import sys
sys.path.append("../..")

import torch

import mods.learning_workflow as lw
import mods.learning_workflow.widgets as lww


def extra_verbose(*_, **__):
    mem_alc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return f"mem: {mem_alc:.1f}/{mem_total:.1f}G({mem_alc / mem_total * 100:.1f}%)"


def main():
    pl = lw.Pipeline(lw.Cfg(10), lw.executor.DummyExecutor())
    pl.register(lww.EtaVerboser(extra_verbose_fn=extra_verbose))

    pl.train(0, 9)


if __name__ == "__main__":
    main()
