import os

import matplotlib.pyplot as plt
import numpy as np
from tools import get_fsct_path


def training_plotter():
    plt.ion()
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)

    while 1:
        try:
            ax1.clear()
            ax2.clear()
            ax1.set_xlabel("Num. Epochs")
            ax1.set_ylabel("Loss")
            ax2.set_xlabel("Num. Epochs")
            ax2.set_ylabel("Accuracy")
            # os.path.join, not string concatenation: get_fsct_path returns the
            # directory without a trailing separator, so the old form asked for
            # ".../modeltraining_history.csv" - a path that never exists. The
            # resulting FileNotFoundError was swallowed by the handler below, so
            # the monitor sat in its loop redrawing an empty figure forever.
            training_history = np.loadtxt(os.path.join(get_fsct_path("model"), "training_history.csv"))
            plt.suptitle("Training History")
            if training_history.shape[0] > 1:
                ax1.plot(training_history[:, 0], training_history[:, 1], c="green", label="train")
                ax1.plot(training_history[:, 0], training_history[:, 3], c="blue", label="validation")
                ax1.legend(loc="lower left")
                ax2.plot(training_history[:, 0], training_history[:, 2], c="green", label="train")
                ax2.plot(training_history[:, 0], training_history[:, 4], c="blue", label="validation")
                ax2.legend(loc="lower left")

            plt.draw()
            plt.pause(120)

        # "except OSError or IndexError" evaluates the boolean first: OSError is
        # truthy, so the handler was really "except OSError" and an IndexError
        # from a short/malformed history file propagated out and killed the
        # monitor. A tuple catches both, which is what was meant.
        except (OSError, IndexError):
            pass


if __name__ == "__main__":
    training_plotter()
