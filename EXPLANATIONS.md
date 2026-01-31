# Performance Take-Home Challenge: A Complete Guide

## Executive Summary

This challenge tasks you with optimizing a kernel that simulates random walks through a binary tree on a custom VLIW SIMD architecture. The baseline implementation runs in **147,734 cycles** - your goal is to reduce this as much as possible by leveraging instruction-level parallelism (packing multiple operations into single cycles) and data-level parallelism (processing 8 elements at once with SIMD). The best known solutions achieve under 1,400 cycles - over 100x faster than the baseline.

---

## Prerequisites Explained from First Principles

### Clock Cycles: The Heartbeat of a Processor

A **clock cycle** is the basic unit of time in a processor. Think of it like a metronome - on each "tick," the processor can perform work. A 3 GHz processor has 3 billion ticks per second.

**Why cycles matter more than wall-clock time:**
- Wall-clock time depends on hardware speed
- Cycle count measures algorithmic efficiency independent of hardware
- In this challenge, fewer cycles = better optimization

### Instruction-Level Parallelism (ILP)

Most code looks sequential:
```
a = b + c      # Step 1
d = e * f      # Step 2
g = a + d      # Step 3 (depends on steps 1 and 2)
```

But steps 1 and 2 are **independent** - they don't need each other's results. A smart processor (or programmer) can run them simultaneously:

```
Cycle 1: [a = b + c] AND [d = e * f]   ← Both execute in parallel
Cycle 2: [g = a + d]                    ← Must wait for cycle 1
```

This reduces 3 sequential cycles to 2 cycles - a 33% speedup from parallelism alone.

### VLIW (Very Long Instruction Word)

**The Analogy: A Factory with Specialized Stations**

Imagine a factory with 5 specialized stations:
- **ALU Station**: Does math (add, multiply, XOR, etc.)
- **VALU Station**: Does math on 8 items at once
- **LOAD Station**: Brings materials from the warehouse
- **STORE Station**: Sends finished products to the warehouse
- **FLOW Station**: Decides what to do next (branching, jumps)

In a traditional processor (like a single worker), you do one task at a time:
```
Cycle 1: Load item A
Cycle 2: Load item B
Cycle 3: Add A + B
Cycle 4: Store result
```

In a **VLIW processor**, all stations work simultaneously on one "instruction word":
```
Cycle 1: [Load A] + [Load B] + [Add X+Y from earlier] + [Store Z from earlier]
```

The "Very Long Instruction Word" packs operations for ALL stations into a single cycle.

**Key insight**: The programmer (you!) must explicitly specify what each station does each cycle. The hardware doesn't figure out parallelism automatically.

### SIMD (Single Instruction Multiple Data)

**The Analogy: One Chef, Eight Carrots**

Imagine chopping carrots:
- **Scalar**: One knife, one carrot at a time → 8 chops for 8 carrots
- **SIMD**: One guillotine, 8 carrots lined up → 1 chop for 8 carrots

In code terms:
```python
# Scalar: 8 separate operations
result[0] = a[0] + b[0]
result[1] = a[1] + b[1]
# ... 6 more times

# SIMD: 1 vector operation
result[0:8] = a[0:8] + b[0:8]  # All 8 happen simultaneously
```

In this architecture, **VLEN = 8**, meaning vector operations process 8 elements at once.

---

## The Architecture in Detail

### The Five Engines

| Engine | Purpose | Slots per Cycle | Key Operations |
|--------|---------|-----------------|----------------|
| **ALU** | Scalar arithmetic | 12 | `+`, `-`, `*`, `//`, `^`, `&`, `\|`, `<<`, `>>`, `%`, `<`, `==` |
| **VALU** | Vector arithmetic | 6 | Same as ALU but on 8 elements, plus `vbroadcast`, `multiply_add` |
| **LOAD** | Read from memory | 2 | `load`, `vload` (8 elements), `const` |
| **STORE** | Write to memory | 2 | `store`, `vstore` (8 elements) |
| **FLOW** | Control flow | 1 | `select`, `vselect`, `jump`, `cond_jump`, `halt` |

**Slot Limits** define how many operations each engine can do per cycle:
```python
SLOT_LIMITS = {
    "alu": 12,    # Up to 12 scalar math ops per cycle
    "valu": 6,    # Up to 6 vector math ops per cycle
    "load": 2,    # Up to 2 loads per cycle
    "store": 2,   # Up to 2 stores per cycle
    "flow": 1,    # Only 1 control flow op per cycle
}
```

### Memory Model

**Two types of storage:**

1. **Main Memory (`mem`)**: Large, slow storage
   - Contains the tree data, input indices, input values
   - Access via `load`/`store` instructions
   - Think of it as RAM

2. **Scratch Space (`scratch`)**: Small, fast storage
   - 1,536 words available
   - Serves as registers and cache
   - All computation happens on scratch values
   - Think of it as registers + L1 cache combined

**Data Flow Rule**: Within a single cycle:
- All **reads** happen at the start of the cycle
- All **writes** happen at the end of the cycle

This means you can read from an address and write to the same address in one cycle - you'll read the old value and write the new value.

### Instruction Format

An instruction is a dictionary mapping engines to lists of operations:

```python
{
    "alu": [
        ("+", dest, src1, src2),     # dest = src1 + src2
        ("*", dest2, src3, src4),    # dest2 = src3 * src4 (parallel!)
    ],
    "load": [
        ("load", dest3, addr_reg),   # dest3 = mem[scratch[addr_reg]]
    ],
    "valu": [
        ("vbroadcast", vdest, scalar_src),  # Copy scalar to all 8 vector elements
    ]
}
```

**Important**: All numbers in instructions are scratch addresses (except for `const` values and jump targets).

---

## The Problem Being Solved

### The Algorithm in Plain English

The kernel performs **random walks through a binary tree**, where:

1. We have a perfect binary tree with values at each node
2. We have a batch of 256 "walkers", each with:
   - A current position (index) in the tree
   - A current value
3. For each of 16 rounds, each walker:
   - Looks up the tree node value at their current position
   - XORs their value with the node value
   - Hashes the result
   - Uses the hash to decide: go left or right in the tree
   - If they fall off the tree, wrap back to the root

### The Hash Function Step by Step

```python
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),  # a = (a + 0x7ED55D16) + (a << 12)
    ("^", 0xC761C23C, "^", ">>", 19),  # a = (a ^ 0xC761C23C) ^ (a >> 19)
    ("+", 0x165667B1, "+", "<<", 5),   # a = (a + 0x165667B1) + (a << 5)
    ("+", 0xD3A2646C, "^", "<<", 9),   # a = (a + 0xD3A2646C) ^ (a << 9)
    ("+", 0xFD7046C5, "+", "<<", 3),   # a = (a + 0xFD7046C5) + (a << 3)
    ("^", 0xB55A4F09, "^", ">>", 16),  # a = (a ^ 0xB55A4F09) ^ (a >> 16)
]
```

Each stage does: `a = op2(op1(a, constant), op3(a, shift_amount))`

For example, stage 1:
```
tmp1 = a + 0x7ED55D16
tmp2 = a << 12
a = tmp1 + tmp2
```

### Memory Layout

```
Address   Content
───────────────────────────────────
0         rounds (16)
1         n_nodes (2047 for height 10)
2         batch_size (256)
3         forest_height (10)
4         forest_values_p → pointer to tree values
5         inp_indices_p → pointer to walker positions
6         inp_values_p → pointer to walker values
7         extra_room_p → scratch area
───────────────────────────────────
[forest_values_p...]   Tree node values (2047 values)
[inp_indices_p...]     Walker positions (256 indices)
[inp_values_p...]      Walker values (256 values)
```

### Algorithm Pseudocode

```python
for round in range(16):           # 16 rounds
    for i in range(256):          # 256 walkers
        idx = indices[i]          # Where am I in the tree?
        val = values[i]           # What's my current value?

        node_val = tree[idx]      # Look up tree node
        val = hash(val ^ node_val)  # XOR and hash

        # Decide direction: left child (2*idx+1) or right child (2*idx+2)
        if val % 2 == 0:
            idx = 2 * idx + 1     # Go left
        else:
            idx = 2 * idx + 2     # Go right

        # Wrap around if we fell off the tree
        if idx >= n_nodes:
            idx = 0

        indices[i] = idx          # Update position
        values[i] = val           # Update value
```

---

## The Baseline Implementation Explained

### Why It's Slow: One Operation Per Cycle

The baseline `build_kernel()` in `perf_takehome.py` creates instructions like:

```python
# Loading walker index
body.append(("alu", ("+", tmp_addr, inp_indices_p, i_const)))  # Cycle 1
body.append(("load", ("load", tmp_idx, tmp_addr)))             # Cycle 2

# Loading walker value
body.append(("alu", ("+", tmp_addr, inp_values_p, i_const)))   # Cycle 3
body.append(("load", ("load", tmp_val, tmp_addr)))             # Cycle 4
```

Each operation gets its own cycle. The `build()` method confirms this:

```python
def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
    instrs = []
    for engine, slot in slots:
        instrs.append({engine: [slot]})  # One slot per instruction!
    return instrs
```

### Cycle Count Breakdown

For 16 rounds × 256 walkers:

| Operation | Cycles per walker | Total |
|-----------|-------------------|-------|
| Load index address calc | 1 | 4,096 |
| Load index | 1 | 4,096 |
| Load value address calc | 1 | 4,096 |
| Load value | 1 | 4,096 |
| Load node address calc | 1 | 4,096 |
| Load node value | 1 | 4,096 |
| XOR | 1 | 4,096 |
| Hash (18 ops × 6 stages) | 18 | 73,728 |
| Direction calculation | 5 | 20,480 |
| Wrap-around check | 2 | 8,192 |
| Store index | 2 | 8,192 |
| Store value | 2 | 8,192 |
| **Total** | ~36 | ~147,456 |

Plus setup overhead → **147,734 cycles**

---

## Optimization Opportunities

### 1. VLIW Instruction Packing

Instead of one operation per cycle, pack multiple independent operations:

```python
# Before: 2 cycles
{"alu": [("+", a, b, c)]}
{"alu": [("*", d, e, f)]}

# After: 1 cycle (if independent)
{"alu": [("+", a, b, c), ("*", d, e, f)]}
```

The hash function has independent sub-computations:
```python
tmp1 = a + constant    # These two can run in parallel!
tmp2 = a << shift      #
a = tmp1 + tmp2        # This must wait
```

### 2. SIMD Vectorization

Process 8 walkers simultaneously instead of 1:

```python
# Before: 8 cycles for 8 walkers
for i in range(8):
    result[i] = a[i] + b[i]

# After: 1 cycle for 8 walkers
{"valu": [("+", result_vec, a_vec, b_vec)]}
```

With VLEN=8, processing 256 walkers becomes 32 vector iterations instead of 256 scalar iterations.

### 3. Loop Unrolling

The baseline fully unrolls all loops (no jumps), which bloats the instruction count but avoids branch penalties. Partial unrolling with actual loops can reduce instruction memory pressure.

### 4. Memory Access Patterns

- **Coalesce loads**: Load 8 contiguous values with `vload` instead of 8 separate `load`s
- **Reduce address calculations**: Reuse computed addresses
- **Prefetching**: Structure loads to hide memory latency

### 5. Theoretical Lower Bound Analysis

Per round, per 8 walkers (one vector):
- Minimum loads: indices, values, node_values = ~3 vloads
- Minimum stores: indices, values = ~2 vstores
- Hash computation: ~18 ops × 6 stages (but many can parallelize)
- With perfect packing: theoretical floor around ~100-200 cycles per round

For 16 rounds × 32 vectors: theoretical minimum ~1000-2000 cycles.

---

## How to Run and Debug

### Key Commands

```bash
# Run correctness and performance test
python perf_takehome.py Tests.test_kernel_cycles

# Generate a trace for visualization
python perf_takehome.py Tests.test_kernel_trace

# Run official submission tests (uses frozen simulator)
python tests/submission_tests.py

# Validate your solution hasn't modified tests
git diff origin/main tests/
```

### Trace Visualization with Perfetto

1. Run the trace test:
   ```bash
   python perf_takehome.py Tests.test_kernel_trace
   ```

2. Start the hot-reload server:
   ```bash
   python watch_trace.py
   ```

3. Open the browser tab and click "Open Perfetto"

4. The trace shows:
   - Each engine's slot utilization over time
   - Which operations execute each cycle
   - Scratch space value changes
   - Empty slots = wasted parallelism opportunities

### Submission Test Thresholds

From `tests/submission_tests.py`:

| Threshold | Achievement |
|-----------|-------------|
| < 147,734 | Beat baseline |
| < 18,532 | Updated starting point |
| < 2,164 | Claude Opus 4 (many hours) |
| < 1,790 | Claude Opus 4.5 casual / best human 2hr |
| < 1,579 | Claude Opus 4.5 (2 hours) |
| < 1,548 | Claude Sonnet 4.5 (many hours) |
| < 1,487 | Claude Opus 4.5 (11.5 hours) |
| < 1,363 | Claude Opus 4.5 improved harness |

---

## Quick Reference

### Instruction Cheat Sheet

```python
# ALU (scalar, up to 12 per cycle)
("op", dest, src1, src2)  # dest = src1 op src2
# ops: +, -, *, //, cdiv, ^, &, |, <<, >>, %, <, ==
# cdiv = ceiling division: (a + b - 1) // b

# VALU (vector, up to 6 per cycle)
("op", dest, src1, src2)      # dest[0:8] = src1[0:8] op src2[0:8]
("vbroadcast", dest, scalar)  # dest[0:8] = [scalar] * 8
("multiply_add", d, a, b, c)  # d[i] = a[i]*b[i] + c[i]

# LOAD (up to 2 per cycle)
("load", dest, addr_reg)              # dest = mem[scratch[addr_reg]]
("load_offset", dest, addr, offset)   # dest+offset = mem[scratch[addr+offset]]
("vload", dest, addr_reg)             # dest[0:8] = mem[addr:addr+8]
("const", dest, value)                # dest = value (immediate)

# STORE (up to 2 per cycle)
("store", addr_reg, src)      # mem[scratch[addr_reg]] = src
("vstore", addr_reg, src)     # mem[addr:addr+8] = src[0:8]

# FLOW (up to 1 per cycle)
("select", d, cond, a, b)     # d = a if cond else b
("add_imm", dest, a, imm)     # dest = a + imm (immediate add)
("vselect", d, cond, a, b)    # vectorized select
("halt",)                     # stop execution
("pause",)                    # pause core (can be resumed)
("trace_write", val)          # append scratch[val] to trace buffer
("cond_jump", cond, addr)     # if cond != 0: pc = addr
("cond_jump_rel", cond, off)  # if cond != 0: pc += off
("jump", addr)                # pc = addr
("jump_indirect", addr)       # pc = scratch[addr]
("coreid", dest)              # dest = core.id
```

### Architecture Constants

```python
VLEN = 8           # Vector length
SCRATCH_SIZE = 1536  # Scratch space words
N_CORES = 1        # Single core (multicore disabled)

# Test parameters
forest_height = 10   # Tree has 2^11 - 1 = 2047 nodes
rounds = 16          # 16 iterations
batch_size = 256     # 256 walkers
```

---

## Summary

The challenge is an exercise in parallel thinking:

1. **Identify independence**: Which operations don't depend on each other?
2. **Pack instructions**: Fill all engine slots each cycle
3. **Use vectors**: Process 8 elements with one instruction
4. **Minimize memory ops**: Loads/stores are precious (only 2 per cycle each)

The gap between 147,734 cycles and <1,400 cycles represents ~100x improvement - achievable through systematic application of these principles.

Good luck!
