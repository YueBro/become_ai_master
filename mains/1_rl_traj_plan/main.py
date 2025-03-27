import sys
sys.path.append("../..")

import mods.learning_workflow as lw


def main():
    pl = lw.Pipeline(lw.Cfg(10), lw.executor.DummyExecutor())

    pl.train(0, 9)


if __name__ == "__main__":
    main()
