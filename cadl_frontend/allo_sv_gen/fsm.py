"""Subtask B: RoCC instruction decode and one FSM chain per CADL instruction.

Statements between two wait states run as nonblocking assignments on the
transition that leaves the earlier state; the first segment runs on the
command-accept edge, as in the manual adapter. On one transition a later
statement sees earlier `let` values by substitution (cast to the declared
width); across states it reads their registers. Each do-while has a head
state, so an iteration costs one extra cycle.

An `if` statement branches the edge; both arms may wait or answer, and the
arms that fall through meet in a join state (one cycle). An SPM element read
`spm[i]` costs a state of its own: the value is registered first, then the
statement that uses it runs. `spm[i] = v` is an assignment on the edge.
Scalar statics are persistent registers: kept across instructions and reset
to their initial value.

A single CPU access `_mem[addr]` (one 32-bit word, byte address) goes through
the HellaCache port with one request in flight: a request state left when the
port takes it, then a response state; a read registers the response data.
A write also waits for its response, so it has landed before the instruction
answers. Fields follow the native flow: size 2, not signed, virtual address.

Runtime checks (SPM indices and ranges, `_mem` alignment) run just before the
access they guard. A failing check aborts the instruction: rd gets 0 and the
cause is OR-ed into a sticky error register, which the reserved instruction
opcode 0x7b funct7 0x7f returns and clears. Earlier side effects stay.

`invoke` and `_mem` DMA statements are delegated to handlers (subtasks C/D),
which add their own states through `FlowBuilder.wait` and `FlowBuilder.drive`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from .. import cadl_ast as ast
from ..allo import constant_integer
from .spm import Spm, SpmLayout

CMD = "rocc_cmd_rocc_cmd_"
RESP = "rocc_resp_rocc_resp_to_bus_"
HELLA = "hella_cmd_hella_cmd_to_bus_"
STATUS = (0x7B, 0x7F)  # reserved: read and clear the error register
ERRORS = {"invoke_range": 1, "spm_index": 2, "dma_range": 4, "mem_align": 8}
FIELDS = ("rs1", "rs2", "rd")
MEMORIES = ("_mem", "_burst_read", "_burst_write")


@dataclass
class Var:
    reg: str
    bits: int
    signed: bool

    def cast(self, value: str) -> str:
        sized = f"{self.bits}'({value})"
        return f"$signed({sized})" if self.signed else sized


@dataclass
class Edge:
    actions: list[str] = field(default_factory=list)
    target: Union["State", "Branch", None] = None


@dataclass
class Branch:
    cond: str
    then: Edge
    other: Edge


@dataclass(eq=False)
class State:
    name: str
    wait: str  # condition to leave the state through `exit`
    exit: Edge = field(default_factory=Edge)


@dataclass
class Decode:
    opcode: int
    funct7: int
    flow: str
    entry: Edge


Handler = Callable[["FlowBuilder", ast.Stmt, dict], None]


class Fsm:
    def __init__(self):
        self.idle = State("S_IDLE", "command_fire")
        self.response = State("S_RESPONSE", f"{RESP}ready", Edge(target=self.idle))
        self.states = [self.idle, self.response]
        self.regs: dict[str, Var] = {}
        self.drives: dict[str, list[str]] = {}
        self.decode: list[Decode] = []
        self.persistent: list[tuple[Var, int]] = []  # scalar statics and their reset values
        self.statics: dict[str, Var] = {}  # proc-level scalar statics
        self.spms: dict[str, Spm] = {}
        self.csrs: set[str] = set()
        self.uses_mem = False  # any single _mem access

    def declare(self, reg: str, bits: int, signed: bool = False) -> Var:
        var = self.regs.setdefault(reg, Var(reg, bits, signed))
        if (var.bits, var.signed) != (bits, signed):
            raise ValueError(f"{reg} redeclared with a different type")
        return var

    def static(self, reg: str, s: ast.Static) -> Var:
        """A persistent register for scalar static `s`, reset to its initial value."""
        if not isinstance(s.ty, ast.DataType_Single):
            raise NotImplementedError(f"static {s.id}: only scalar statics outside SPMs")
        init = 0 if s.expr is None else constant_integer(s.expr)
        if init is None:
            raise ValueError(f"static {s.id}: initial value must be an integer constant")
        var = self.declare(reg, *_basic(s.ty.basic_type))
        self.persistent.append((var, init))
        return var


def _basic(ty: ast.BasicType) -> tuple[int, bool]:
    if isinstance(ty, ast.BasicType_ApUFixed):
        return ty.width, False
    if isinstance(ty, ast.BasicType_ApFixed):
        return ty.width, True
    raise NotImplementedError(f"FSM variables of type {ty}")


def _is_irf(e: ast.Expr) -> bool:
    return (isinstance(e, ast.IndexExpr) and isinstance(e.expr, ast.IdentExpr)
            and e.expr.name == "_irf" and len(e.indices) == 1
            and isinstance(e.indices[0], ast.IdentExpr))


def _is_dma(s: ast.AssignStmt) -> bool:
    return any(isinstance(side, ast.RangeSliceExpr) and isinstance(side.expr, ast.IdentExpr)
               and side.expr.name in MEMORIES for side in (s.lhs, s.rhs))


_UNARY = {ast.UnaryOp.NEG: "-", ast.UnaryOp.NOT: "!", ast.UnaryOp.BIT_NOT: "~"}
_CASTS = {ast.UnaryOp.SIGNED_CAST: "$signed", ast.UnaryOp.UNSIGNED_CAST: "$unsigned"}


class FlowBuilder:
    def __init__(self, fsm: Fsm, flow: ast.Flow, handlers: dict[str, Handler]):
        self.fsm, self.flow, self.handlers = fsm, flow, handlers
        self.entry = self.edge = Edge()
        self.env: dict[str, str] = {}  # values assigned on the current edge
        self.scope: dict[str, Var] = dict(fsm.statics)
        self.on_fire = True
        self.count = 0
        self.done = False
        self.loaded: dict[int, str] = {}  # id(SPM element read) -> register holding it

    # Interface for the invoke (C) and DMA (D) handlers.

    def action(self, line: str) -> None:
        self.edge.actions.append(line)

    def wait(self, tag: str, cond: str) -> State:
        """Close the current edge into a new state left when `cond` holds."""
        self.count += 1
        state = State(f"S_{self.flow.name}_{tag}{self.count}", cond)
        self.fsm.states.append(state)
        self._goto(state)
        self.edge, self.env, self.on_fire = state.exit, {}, False
        return state

    def wait_unless(self, tag: str, cond: str) -> State:
        """Like `wait`, but go straight on when `cond` already holds on this edge.

        Both ways share the continuing edge, which is emitted at each of them.
        """
        self.count += 1
        rest = Edge()
        state = State(f"S_{self.flow.name}_{tag}{self.count}", cond, rest)
        self.fsm.states.append(state)
        self._goto(Branch(cond, rest, Edge(target=state)))
        self.edge, self.env, self.on_fire = rest, {}, False
        return state

    def drive(self, signal: str, state: State) -> None:
        """Hold `signal` high while the FSM is in `state`."""
        self.fsm.drives.setdefault(signal, []).append(state.name)

    def declare(self, name: str, bits: int, signed: bool = False) -> Var:
        return self.fsm.declare(f"{self.flow.name}__{name}", bits, signed)

    def expr(self, e: ast.Expr) -> str:
        if isinstance(e, ast.LitExpr):
            if not isinstance(e.literal.lit, ast.LiteralInner_Fixed):
                raise NotImplementedError(f"{self.flow.name}: literal {e}")
            bits, signed = _basic(e.literal.ty)
            value = e.literal.lit.value
            return f"{bits}'{'s' if signed else ''}d{value}" if value >= 0 else f"-{bits}'sd{-value}"
        if isinstance(e, ast.IdentExpr):
            if e.name in self.scope:
                return self.env.get(e.name, self.scope[e.name].reg)
            if e.name in self.fsm.csrs:
                raise NotImplementedError(f"{self.flow.name}: CSR register {e.name}")
            if e.name in FIELDS and any(name == e.name for name, _ in self.flow.inputs):
                return self._fire(f"{CMD}{e.name}")
            raise ValueError(f"{self.flow.name}: unknown name {e.name}")
        if _is_irf(e):
            index = e.indices[0].name
            if index not in ("rs1", "rs2"):
                raise ValueError(f"{self.flow.name}: _irf read must use rs1 or rs2")
            return self._fire(f"{CMD}{index}data")
        if self._spm(e) or self._cpu(e):
            if id(e) not in self.loaded:
                raise NotImplementedError(f"{self.flow.name}: element read in {e} here")
            return self.env.get(f"#load{id(e)}", self.loaded[id(e)])
        if isinstance(e, ast.BinaryExpr):
            return f"({self.expr(e.left)} {e.op.value} {self.expr(e.right)})"
        if isinstance(e, ast.UnaryExpr) and e.op in _UNARY:
            return f"({_UNARY[e.op]}{self.expr(e.operand)})"
        if isinstance(e, ast.UnaryExpr) and e.op in _CASTS:
            return f"{_CASTS[e.op]}({self.expr(e.operand)})"
        if isinstance(e, ast.IfExpr):
            return (f"({self.expr(e.condition)} ? {self.expr(e.then_branch)}"
                    f" : {self.expr(e.else_branch)})")
        raise NotImplementedError(f"{self.flow.name}: expression {e}")

    def _spm(self, e: ast.Expr) -> Spm | None:
        """The SPM of an element access `spm[i]`, if `e` is one."""
        if (isinstance(e, ast.IndexExpr) and isinstance(e.expr, ast.IdentExpr)
                and e.expr.name in self.fsm.spms and e.expr.name not in self.scope):
            if len(e.indices) != 1:
                raise NotImplementedError(f"{self.flow.name}: multi-index SPM access {e}")
            return self.fsm.spms[e.expr.name]
        return None

    def _cpu(self, e: ast.Expr) -> bool:
        """Whether `e` is a single CPU memory access `_mem[addr]`."""
        if (isinstance(e, ast.IndexExpr) and isinstance(e.expr, ast.IdentExpr)
                and e.expr.name == "_mem" and "_mem" not in self.scope):
            if len(e.indices) != 1:
                raise NotImplementedError(f"{self.flow.name}: multi-index _mem access {e}")
            return True
        return False

    def check(self, cond: str, cause: str) -> None:
        """Abort the instruction with error `cause` if `cond` holds on this edge."""
        bad = Edge([f"error_status <= error_status | 32'd{ERRORS[cause]};",
                    "response_data <= 32'd0;"], self.fsm.response)
        ok = Edge()
        self._goto(Branch(cond, bad, ok))
        self.edge = ok

    def check_range(self, start: ast.Expr, count: int, spm: Spm, cause: str, what: str) -> None:
        """Elements [start, start + count) must lie in `spm`: at build time if constant."""
        first = constant_integer(start)
        if first is None:
            self.check(f"(33'({self.expr(start)}) + 33'd{count}) > 33'd{spm.elements}", cause)
        elif first < 0 or first + count > spm.elements:
            raise ValueError(f"{self.flow.name}: {what} [{first}, {first + count}) "
                             f"outside {spm.name}'s {spm.elements} elements")

    def _request(self, addr: ast.Expr, write: bool, data: str = "32'd0") -> None:
        """One HellaCache access: wait until the port takes it, then for the response."""
        self.fsm.uses_mem = True
        offset = constant_integer(addr)
        if offset is None:
            self.check(f"(({self.expr(addr)}) & 32'd3) != 32'd0", "mem_align")
        elif offset % 4:
            raise ValueError(f"{self.flow.name}: _mem[{offset}] is not 4-byte aligned")
        addr = self.expr(addr)
        self.action(f"mem_addr <= 32'({addr});")
        self.action(f"mem_write <= 1'b{int(write)};")
        self.action(f"mem_data <= 32'({data});")
        self.drive(f"{HELLA}enable", self.wait("mem_req", f"{HELLA}ready"))
        self.wait("mem_resp", "hella_resp_enable")

    def load(self, *exprs: ast.Expr | None) -> None:
        """Register every element read (SPM or _mem) in `exprs`, inner indices first."""
        for e in exprs:
            if e is None:
                continue
            spm = self._spm(e)
            if spm is not None:
                self.load(e.indices[0])
                if id(e) not in self.loaded:
                    self.check_range(e.indices[0], 1, spm, "spm_index", f"{spm.name} index")
                    var = self.declare(f"spm_read{len(self.loaded)}", spm.elem_bits, spm.signed)
                    self.action(f"{var.reg} <= {spm.signal}[{self.expr(e.indices[0])}];")
                    self.wait("spm_read", "1'b1")
                    self.loaded[id(e)] = var.reg
                continue
            if self._cpu(e):
                self.load(e.indices[0])
                if id(e) not in self.loaded:
                    var = self.declare(f"mem_read{len(self.loaded)}", 32)
                    self._request(e.indices[0], write=False)
                    self.action(f"{var.reg} <= hella_resp_hella_resp_data;")
                    self.loaded[id(e)] = var.reg
                    # Statements on the response edge read the port, not the register.
                    self.env[f"#load{id(e)}"] = "hella_resp_hella_resp_data"
                continue
            for child in vars(e).values() if hasattr(e, "__dict__") else ():
                for item in child if isinstance(child, list) else [child]:
                    if isinstance(item, ast.Expr):
                        self.load(item)

    # Statement translation.

    def run(self, body: list[ast.Stmt]) -> None:
        directives: dict[str, Optional[ast.Expr]] = {}
        for stmt in body:
            if self.done:
                raise ValueError(f"{self.flow.name}: statement after the _irf[rd] write")
            if isinstance(stmt, ast.DirectiveStmt):
                directives[stmt.name] = stmt.expr
                continue
            self._stmt(stmt, directives)
            directives = {}
        if directives:
            raise ValueError(f"{self.flow.name}: directives {list(directives)} apply to nothing")

    def finish(self) -> None:
        if not self.done:
            self._goto(self.fsm.idle)

    def _stmt(self, s: ast.Stmt, directives: dict) -> None:
        if isinstance(s, ast.InvokeStmt):
            return self._handle("invoke", s, directives)
        if isinstance(s, ast.AssignStmt) and _is_dma(s):
            return self._handle("dma", s, directives)
        if directives:
            raise NotImplementedError(f"{self.flow.name}: directives on {s}")
        if isinstance(s, ast.AssignStmt) and _is_irf(s.lhs):
            return self._respond(s)
        if isinstance(s, ast.AssignStmt) and isinstance(s.lhs, ast.IdentExpr):
            return self._assign(s)
        if isinstance(s, ast.AssignStmt) and self._spm(s.lhs):
            return self._store(s)
        if isinstance(s, ast.AssignStmt) and self._cpu(s.lhs):
            self.load(s.lhs.indices[0], s.rhs)
            return self._request(s.lhs.indices[0], True, self.expr(s.rhs))
        if isinstance(s, ast.DoWhileStmt):
            return self._loop(s)
        if isinstance(s, ast.IfStmt):
            return self._if(s)
        if isinstance(s, ast.StaticStmt):
            self.scope[s.static.id] = self.fsm.static(f"{self.flow.name}__{s.static.id}", s.static)
            return None
        raise NotImplementedError(f"{self.flow.name}: statement {s}")

    def _handle(self, kind: str, s: ast.Stmt, directives: dict) -> None:
        handler = self.handlers.get(kind)
        if handler is None:
            raise NotImplementedError(f"{self.flow.name}: no handler for {kind} statements")
        handler(self, s, directives)

    def _assign(self, s: ast.AssignStmt) -> None:
        name = s.lhs.name
        self.load(s.rhs)
        value = self.expr(s.rhs)
        if s.is_let:
            if not isinstance(s.type_annotation, ast.DataType_Single):
                raise ValueError(f"{self.flow.name}: let {name} needs a scalar type annotation")
            self.scope[name] = self.declare(name, *_basic(s.type_annotation.basic_type))
        elif name not in self.scope:
            raise ValueError(f"{self.flow.name}: assignment to undeclared {name}")
        var = self.scope[name]
        self.action(f"{var.reg} <= {var.cast(value)};")
        self.env[name] = var.cast(value)

    def _respond(self, s: ast.AssignStmt) -> None:
        if s.lhs.indices[0].name != "rd":
            raise ValueError(f"{self.flow.name}: _irf writes must target rd")
        self.load(s.rhs)
        self.action(f"response_data <= 32'({self.expr(s.rhs)});")
        self._goto(self.fsm.response)
        self.done = True

    def _store(self, s: ast.AssignStmt) -> None:
        spm = self._spm(s.lhs)
        self.load(s.lhs.indices[0], s.rhs)
        self.check_range(s.lhs.indices[0], 1, spm, "spm_index", f"{spm.name} index")
        self.action(f"{spm.signal}[{self.expr(s.lhs.indices[0])}] <= "
                    f"{spm.elem_bits}'({self.expr(s.rhs)});")

    def _if(self, s: ast.IfStmt) -> None:
        self.load(s.condition)
        cond = self.expr(s.condition)
        arms = (Edge(), Edge())
        self._goto(Branch(cond, *arms))
        scope, env, on_fire = dict(self.scope), dict(self.env), self.on_fire
        open_ends = []
        for body, edge in zip((s.then_body, s.else_body or []), arms):
            self.edge, self.env, self.scope, self.on_fire = edge, dict(env), dict(scope), on_fire
            self.done = False
            self.run(body)
            if not self.done:
                open_ends.append(self.edge)
        self.scope = scope
        if not open_ends:  # both arms answered
            self.done = True
            return
        self.count += 1
        join = State(f"S_{self.flow.name}_join{self.count}", "1'b1")
        self.fsm.states.append(join)
        for edge in open_ends:
            edge.target = join
        self.edge, self.env, self.on_fire = join.exit, {}, False

    def _loop(self, s: ast.DoWhileStmt) -> None:
        outer = dict(self.scope)
        if any(b.init is None for b in s.bindings):
            raise ValueError(f"{self.flow.name}: loop bindings need an initial value")
        self.load(*(b.init for b in s.bindings))
        inits = [self.expr(b.init) for b in s.bindings]  # all read pre-loop values
        loop_vars = [self.declare(b.id, *_basic(b.ty)) for b in s.bindings]
        for b, var, init in zip(s.bindings, loop_vars, inits):
            self.scope[b.id] = var
            self.action(f"{var.reg} <= {var.cast(init)};")
        head = self.wait("loop", "1'b1")
        self.run(s.body)
        if self.done:
            raise ValueError(f"{self.flow.name}: _irf[rd] write inside a loop")
        self.load(*(b.next for b in s.bindings), s.condition)
        nexts = [var.cast(self.expr(b.next)) if b.next else var.reg
                 for b, var in zip(s.bindings, loop_vars)]
        cond = self.expr(s.condition)
        for var, value in zip(loop_vars, nexts):
            self.action(f"{var.reg} <= {value};")
        rest = Edge()
        self._goto(Branch(cond, Edge(target=head), rest))
        self.edge = rest
        self.scope = outer | {b.id: var for b, var in zip(s.bindings, loop_vars)}
        # Outer variables assigned on this edge keep their pending values.
        self.env = {k: v for k, v in self.env.items() if k in outer or k.startswith("#load")}
        self.env |= {b.id: value for b, value in zip(s.bindings, nexts)}

    def _fire(self, signal: str) -> str:
        if not self.on_fire:
            raise ValueError(f"{self.flow.name}: {signal} is only valid before the first wait state")
        return signal

    def _goto(self, target: Union[State, Branch]) -> None:
        assert self.edge.target is None
        self.edge.target = target


def _attr(flow: ast.Flow, name: str) -> int:
    value = flow.attrs.get(name)
    if not (isinstance(value, ast.LitExpr) and isinstance(value.literal.lit, ast.LiteralInner_Fixed)):
        raise ValueError(f"{flow.name}: missing #[{name}(...)]")
    return value.literal.lit.value


def build_fsm(proc: ast.Proc, handlers: dict[str, Handler],
              layout: SpmLayout | None = None) -> Fsm:
    fsm = Fsm()
    fsm.spms = {spm.name: spm for spm in layout.spms} if layout else {}
    fsm.csrs = set(proc.registers)
    for name, static in proc.statics.items():
        if not isinstance(static.ty, ast.DataType_Array):
            fsm.statics[name] = fsm.static(f"static_{name}", static)
    seen: dict[tuple[int, int], str] = {STATUS: "the reserved error status instruction"}
    for flow in proc.flows.values():
        if flow.kind != ast.FlowKind.RTYPE:
            raise NotImplementedError(f"{flow.name}: only rtype instructions are supported")
        key = (_attr(flow, "opcode"), _attr(flow, "funct7"))
        if key in seen:
            raise ValueError(f"{flow.name} and {seen[key]} share opcode/funct7 {key}")
        seen[key] = flow.name
        builder = FlowBuilder(fsm, flow, handlers)
        builder.run(flow.body or [])
        builder.finish()
        fsm.decode.append(Decode(*key, flow.name, builder.entry))
    return fsm


def _edge(edge: Edge, indent: str) -> list[str]:
    lines = [indent + action for action in edge.actions]
    target = edge.target
    if isinstance(target, State):
        lines.append(f"{indent}state <= {target.name};")
    elif isinstance(target, Branch):
        lines.append(f"{indent}if ({target.cond}) begin")
        lines += _edge(target.then, indent + "  ")
        lines.append(f"{indent}end else begin")
        lines += _edge(target.other, indent + "  ")
        lines.append(f"{indent}end")
    else:
        raise AssertionError("open FSM edge")
    return lines


def generate_fsm(fsm: Fsm, port_names: set[str]) -> str:
    width = max(1, math.ceil(math.log2(len(fsm.states))))
    names = ",\n  ".join(f"{s.name} = {i}" for i, s in enumerate(fsm.states))
    out = [f"localparam [{width - 1}:0]\n  {names};",
           f"reg [{width - 1}:0] state;", "reg [4:0] saved_rd;", "reg [31:0] response_data;",
           "reg [31:0] error_status;"]
    out += [f"reg {'signed ' if v.signed else ''}[{v.bits - 1}:0] {v.reg};"
            for v in fsm.regs.values()]
    if fsm.uses_mem:
        out += ["reg [31:0] mem_addr, mem_data;", "reg mem_write;",
                f"assign {HELLA}cmd_addr = mem_addr;", f"assign {HELLA}cmd_tag = 8'd0;",
                f"assign {HELLA}cmd_cmd = {{4'd0, mem_write}};", f"assign {HELLA}cmd_size = 2'd2;",
                f"assign {HELLA}cmd_signed = 1'b0;", f"assign {HELLA}cmd_phys = 1'b0;",
                f"assign {HELLA}cmd_data = mem_data;", f"assign {HELLA}cmd_mask = 4'd0;",
                "assign hella_resp_ready = 1'b1;"]
    out += ["wire command_fire = rocc_cmd_enable && rocc_cmd_ready;",
            "assign rocc_cmd_ready = !rst && state == S_IDLE;",
            f"assign {RESP}enable = !rst && state == S_RESPONSE;",
            f"assign {RESP}result_rd = saved_rd;",
            f"assign {RESP}result_rddata = response_data;"]
    for signal, states in fsm.drives.items():
        value = "!rst && (" + " || ".join(f"state == {s}" for s in states) + ")"
        out.append(f"assign {signal} = {value};" if signal in port_names
                   else f"always @(*) {signal} = {value};")

    body = ["always @(posedge clk) begin",
            "  if (rst) begin",
            "    state <= S_IDLE;",
            "    saved_rd <= 0;",
            "    response_data <= 0;",
            "    error_status <= 0;"]
    body += [f"    {var.reg} <= {var.cast(str(init) if init >= 0 else f'-{-init}')};"
             for var, init in fsm.persistent]
    body += ["  end else begin",
            "    case (state)",
            "      S_IDLE: if (command_fire) begin",
            f"        saved_rd <= {CMD}rd;",
            "        response_data <= 0;",
            f"        case ({{{CMD}opcode, {CMD}funct}})"]
    for d in fsm.decode:
        body.append(f"          {{7'h{d.opcode:02x}, 7'h{d.funct7:02x}}}: begin  // {d.flow}")
        body += _edge(d.entry, " " * 12)
        body.append("          end")
    body += [f"          {{7'h{STATUS[0]:02x}, 7'h{STATUS[1]:02x}}}: begin  // error status",
             "            response_data <= error_status;",
             "            error_status <= 32'd0;",
             "            state <= S_RESPONSE;",
             "          end"]
    body += ["          default: begin",
             "            response_data <= 32'd1;",
             "            state <= S_RESPONSE;",
             "          end",
             "        endcase",
             "      end"]
    for state in fsm.states[1:]:
        body.append(f"      {state.name}: if ({state.wait}) begin")
        body += _edge(state.exit, " " * 8)
        body.append("      end")
    body += ["      default: state <= S_IDLE;", "    endcase", "  end", "end"]
    return "\n".join(out + body) + "\n"
