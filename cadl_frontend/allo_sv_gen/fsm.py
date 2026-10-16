"""Subtask B: RoCC instruction decode and one FSM chain per CADL instruction.

Statements between two wait states run as nonblocking assignments on the
transition that leaves the earlier state; the first segment runs on the
command-accept edge, as in the manual adapter. On one transition a later
statement sees earlier `let` values by substitution (cast to the declared
width); across states it reads their registers. Each do-while has a head
state, so an iteration costs one extra cycle.

`invoke` and `_mem` DMA statements are delegated to handlers (subtasks C/D),
which add their own states through `FlowBuilder.wait` and `FlowBuilder.drive`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from .. import cadl_ast as ast

CMD = "rocc_cmd_rocc_cmd_"
RESP = "rocc_resp_rocc_resp_to_bus_"
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

    def declare(self, reg: str, bits: int, signed: bool = False) -> Var:
        var = self.regs.setdefault(reg, Var(reg, bits, signed))
        if (var.bits, var.signed) != (bits, signed):
            raise ValueError(f"{reg} redeclared with a different type")
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
        self.scope: dict[str, Var] = {}
        self.on_fire = True
        self.count = 0
        self.done = False

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
            if e.name in FIELDS and any(name == e.name for name, _ in self.flow.inputs):
                return self._fire(f"{CMD}{e.name}")
            raise ValueError(f"{self.flow.name}: unknown name {e.name}")
        if _is_irf(e):
            index = e.indices[0].name
            if index not in ("rs1", "rs2"):
                raise ValueError(f"{self.flow.name}: _irf read must use rs1 or rs2")
            return self._fire(f"{CMD}{index}data")
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
        if isinstance(s, ast.DoWhileStmt):
            return self._loop(s)
        raise NotImplementedError(f"{self.flow.name}: statement {s}")

    def _handle(self, kind: str, s: ast.Stmt, directives: dict) -> None:
        handler = self.handlers.get(kind)
        if handler is None:
            raise NotImplementedError(f"{self.flow.name}: no handler for {kind} statements")
        handler(self, s, directives)

    def _assign(self, s: ast.AssignStmt) -> None:
        name = s.lhs.name
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
        self.action(f"response_data <= 32'({self.expr(s.rhs)});")
        self._goto(self.fsm.response)
        self.done = True

    def _loop(self, s: ast.DoWhileStmt) -> None:
        outer = dict(self.scope)
        if any(b.init is None for b in s.bindings):
            raise ValueError(f"{self.flow.name}: loop bindings need an initial value")
        inits = [self.expr(b.init) for b in s.bindings]  # all read pre-loop values
        loop_vars = [self.declare(b.id, *_basic(b.ty)) for b in s.bindings]
        for b, var, init in zip(s.bindings, loop_vars, inits):
            self.scope[b.id] = var
            self.action(f"{var.reg} <= {var.cast(init)};")
        head = self.wait("loop", "1'b1")
        self.run(s.body)
        if self.done:
            raise ValueError(f"{self.flow.name}: _irf[rd] write inside a loop")
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
        self.env = {k: v for k, v in self.env.items() if k in outer}
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


def build_fsm(proc: ast.Proc, handlers: dict[str, Handler]) -> Fsm:
    fsm = Fsm()
    seen: dict[tuple[int, int], str] = {}
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
           f"reg [{width - 1}:0] state;", "reg [4:0] saved_rd;", "reg [31:0] response_data;"]
    out += [f"reg {'signed ' if v.signed else ''}[{v.bits - 1}:0] {v.reg};"
            for v in fsm.regs.values()]
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
            "  end else begin",
            "    case (state)",
            "      S_IDLE: if (command_fire) begin",
            f"        saved_rd <= {CMD}rd;",
            "        response_data <= 0;",
            f"        case ({{{CMD}opcode, {CMD}funct}})"]
    for d in fsm.decode:
        body.append(f"          {{7'h{d.opcode:02x}, 7'h{d.funct7:02x}}}: begin  // {d.flow}")
        body += _edge(d.entry, " " * 12)
        body.append("          end")
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
