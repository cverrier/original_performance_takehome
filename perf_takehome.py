"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple] | dict[Engine, list[tuple]]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        # Also handles pre-built instruction bundles (dicts) for parallel operations
        instrs = []
        for item in slots:
            if isinstance(item, dict):
                # Already a pre-built instruction bundle
                instrs.append(item)
            else:
                engine, slot = item
                instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def init_hash_constants(self) -> tuple[list, list]:
        """Pre-load hash constants into scratch addresses for use in build_hash."""
        hash_val1_addrs = []
        hash_val3_addrs = []
        for _, val1, _, _, val3 in HASH_STAGES:
            val1_addr = self.scratch_const(val1)
            val3_addr = self.scratch_const(val3)
            hash_val1_addrs.append(val1_addr)
            hash_val3_addrs.append(val3_addr)
        return hash_val1_addrs, hash_val3_addrs

    def build_hash(self, val_hash_addr, hash_val1_vecs, hash_val3_vecs, tmp1, tmp2, round, i):
        slots = []

        for hi, ((op1, _, op2, op3, _), hash_val1_vec, hash_val3_vec) in enumerate(zip(HASH_STAGES, hash_val1_vecs, hash_val3_vecs)):
            # Execute the two independent ALU operations in parallel (one cycle)
            slots.append({"valu": [
                (op1, tmp1, val_hash_addr, hash_val1_vec),
                (op3, tmp2, val_hash_addr, hash_val3_vec)
            ]})
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("vcompare", val_hash_addr, [(round, i+j, "hash_stage", hi) for j in range(VLEN)])))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1", length=VLEN)
        tmp2 = self.alloc_scratch("tmp2", length=VLEN)
        tmp3 = self.alloc_scratch("tmp3", length=VLEN)
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        n_nodes_vec = self.alloc_scratch("n_nodes_vec", length=VLEN)
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        # Allocate and initialize an offset vector so we can process VLEN
        # workers in parallel.
        # offset_vec = [0, 1, 2, ..., VLEN - 1]
        # NOTE: Based on how 'vload' works, this approach is not needed when
        # loading contiguous elements. However, we keep this commented out for
        # now, as it might be useful to load/store when things aren't contiguous
        # offset_vec = self.alloc_scratch("offset_vec", length=VLEN)
        # for i in range(VLEN):
        #     self.add("load", ("const", offset_vec + i, i))

        zero_const = self.scratch_const(0)
        zero_const_vec = self.alloc_scratch("zero_const_vec", length=VLEN)
        self.add("valu", ("vbroadcast", zero_const_vec, zero_const))
        one_const = self.scratch_const(1)
        one_const_vec = self.alloc_scratch("one_const_vec", length=VLEN)
        self.add("valu", ("vbroadcast", one_const_vec, one_const))
        two_const = self.scratch_const(2)
        two_const_vec = self.alloc_scratch("two_const_vec", length=VLEN)
        self.add("valu", ("vbroadcast", two_const_vec, two_const))

        # Pre-load hash constants and broadcast them
        hash_val1_addrs, hash_val3_addrs = self.init_hash_constants()
        val1_vecs = []
        val3_vecs = []
        for hi in range(len(HASH_STAGES)):
            val1_hi_vec = self.alloc_scratch(f"val1_{hi}_vec", length=VLEN)
            val3_hi_vec = self.alloc_scratch(f"val3_{hi}_vec", length=VLEN)
            val1_hi_addr = hash_val1_addrs[hi]
            val3_hi_addr = hash_val3_addrs[hi]
            self.instrs.append({"valu": [
                ("vbroadcast", val1_hi_vec, val1_hi_addr),
                ("vbroadcast", val3_hi_vec, val3_hi_addr)
            ]})
            val1_vecs.append(val1_hi_vec)
            val3_vecs.append(val3_hi_vec)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scratch registers for loop body
        tmp_idx = self.alloc_scratch("tmp_idx", length=VLEN)
        tmp_val = self.alloc_scratch("tmp_val", length=VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", length=VLEN)
        tmp_addr = self.alloc_scratch("tmp_addr", length=1)
        tmp_addr_vec = self.alloc_scratch("tmp_addr_vec", length=VLEN)

        tmp_inp_indices_addr = self.alloc_scratch("tmp_inp_indices_addr", length=1)
        tmp_inp_values_addr = self.alloc_scratch("tmp_inp_values_addr", length=1)

        # Broadcast forest_values_p for gather operation
        forest_values_p_vec = self.alloc_scratch("forest_values_p_vec", length=VLEN)
        self.add("valu", ("vbroadcast", forest_values_p_vec, self.scratch["forest_values_p"]))

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                # Load node indices and input values in parallel
                body.append({"alu": [
                    ("+", tmp_inp_indices_addr, self.scratch["inp_indices_p"], i_const),
                    ("+", tmp_inp_values_addr, self.scratch["inp_values_p"], i_const)
                ]})
                body.append({"load": [
                    ("vload", tmp_idx, tmp_inp_indices_addr),
                    ("vload", tmp_val, tmp_inp_values_addr)
                ]})
                body.append(("debug", ("vcompare", tmp_idx, [(round, i+j, "idx") for j in range(VLEN)])))
                body.append(("debug", ("vcompare", tmp_val, [(round, i+j, "val") for j in range(VLEN)])))
                # Load node values (gather: node_val[j] = mem[forest_values_p + idx[j]])
                body.append(("valu", ("+", tmp_addr_vec, forest_values_p_vec, tmp_idx)))
                for j in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_addr_vec, j)))
                body.append(("debug", ("vcompare", tmp_node_val, [(round, i+j, "node_val") for j in range(VLEN)])))
                # Compute XOR and hash values
                body.append(("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, val1_vecs, val3_vecs, tmp1, tmp2, round, i))
                body.append(("debug", ("vcompare", tmp_val, [(round, i+j, "hashed_val") for j in range(VLEN)])))
                # Compute idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1, tmp_val, two_const_vec)))
                body.append(("valu", ("==", tmp1, tmp1, zero_const_vec)))
                body.append(("flow", ("vselect", tmp3, tmp1, one_const_vec, two_const_vec)))
                body.append(("valu", ("*", tmp_idx, tmp_idx, two_const_vec)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("vcompare", tmp_idx, [(round, i+j, "next_idx") for j in range(VLEN)])))
                # Compute idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", tmp1, tmp_idx, n_nodes_vec)))
                body.append(("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_const_vec)))
                body.append(("debug", ("vcompare", tmp_idx, [(round, i+j, "wrapped_idx") for j in range(VLEN)])))
                # Compute mem[inp_indices_p + i] = idx
                # TODO: Keep this address since we already compute it at the beginning
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, tmp_idx)))
                # Compute mem[inp_values_p + i] = val
                # TODO: Keep this address since we already compute it at the beginning
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
