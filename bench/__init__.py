"""ClauDali's benchmark harness.

Two experiments that share one runner, one state file and one report:

* the **settings benchmark**, which holds the picture still and varies the
  tool's options, to find out what each one costs and buys; and
* the **topic survey**, which holds the options still and varies the picture,
  to find out what the models are good at.

Mixing them would confound both, so they are separate phases of one ordered
list. See ``bench/README.md`` for how to run it.

Nothing in this package imports torch except :mod:`bench.worker`, which is the
only module that renders. ``bench plan``, ``bench status`` and ``bench report``
stay fast because of it.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
