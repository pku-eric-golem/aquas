"""Assemble the APS `main` module from the route 1 sub-generators."""
from __future__ import annotations

from .. import cadl_ast as ast
from .burst import generate_burst
from .dma import DmaBridge
from .fsm import build_fsm, generate_fsm
from ..allo_kernel.interface import HlsInstance
from .instantiate import generate_hls_instance
from .invoke import InvokeBridge
from .ports import WrapperPort, assemble_module, load_ports
from .spm import build_layout, generate_spm


def generate_main(proc: ast.Proc, instance: HlsInstance,
                  ports: list[WrapperPort] | None = None) -> str:
    ports = ports or load_ports()
    layout = build_layout(proc)
    invoke, dma = InvokeBridge(instance, layout), DmaBridge(layout)
    fsm = build_fsm(proc, {"invoke": invoke.handle, "dma": dma.handle})
    # Declarations come before the FSM and bridges that use them.
    body = (generate_spm(layout) + generate_hls_instance(instance) + invoke.declarations()
            + dma.declarations() + generate_fsm(fsm, {p.name for p in ports})
            + invoke.bridge() + dma.wiring() + generate_burst(layout, ports))
    return assemble_module(ports, body)
