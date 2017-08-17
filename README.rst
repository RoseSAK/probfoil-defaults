Abnormal Predicates: Learning Categorical Defaults from Probabilistic Rules
MSc Project - Rose Azad Khan
========================
This repository contains the original ProbFOIL code, along with several files and folders created for use in this MSc project.

In the probfoil folder: every file in this folder belongs to the original ProbFOIL implementation, except for defaults.py,
defaults_complex.py and generator.py. The original ProbFOIL files are unmodified except for probfoil.py

defaults.py contains the function construct_ab_pred, which creates the abnormal predicate and new data points and writes these to the
data files.

defaults_complex.py is identical to defaults.py except that it includes a loop which activates when there is more than one positive predicate
in the body of the rule i.e. more than one candidate for creating ab_pred. This file was used when complex examples were explored in the
qualitative evaluation

generator.py is the program used to generate datasets for the quantitative evaluation. It can be run from the command line, you need to specify
an output data file <file.data> and an output settings file <file.settings.pl> and the number of objects in the domain N

probfoil.py has been modified to learn categorical defaults rather than statistical defaults. At the moment, probfoil.py uses the function
from defaults_complex.py (line 459 of probfoil.py). If you wish to use the function from defaults.py instead, you must change the import statement at the top of the
file to 'import defaults' instead of import defaults_complex. The function names are identical

To run the modified probfoil algorithm, you need to use the repository version of probfoil. If you are using files from a separate folder,
you will need to run probfoil as a module, due to importing
The command is 'python -m probfoil.probfoil -p 0.0 -v <file.settings.pl> <file.data> '
The significance value p must be set to 0.0. The verbose setting is optional (-v, -vv, -vvv etc)

The folder 'datasets' contains the datasets which were generated for the quantitative evaluation, along with a file analysis.py which was
used to plot the quantitative results on graphs

The folder 'examples' contains a large number of data files and settings file, which were used for experimentation and for learning complex
examples.

The folder 'alchemy-files' contains example Alchemy .mln and .db files, from the initial experimentation with Alchemy

Running the modified probfoil algorithm writes data points to both the data file and the settings file. If you wish to run the algorithm
more than once using the same files, you will need to open the files and delete the new data points, unless you wish to keep the new
abnormal predicate for the next learning stage

The text below is the original ProbFOIL readme

ProbFOIL v2.1
=============

ProbFOIL is a probabilistic extension of FOIL that is capable of learning probabilistic rules from
probabilistic data.

ProbFOIL 2.1 is a redesign of the Prob2FOIL algorithm that was introduced in https://lirias.kuleuven.be/handle/123456789/499989.
It works on top of ProbLog 2.1

If you are looking for the version used in the paper, you should check out the tag ``paper_version``.

Installation
------------

ProbFOIL 2.1 requires ProbLog 2.1.
You can install ProbLog by using the command:

.. code-block:: python

    pip install problog

ProbFOIL does not require any further installation.

Usage
-----

The input of ProbFOIL consists of two parts: settings and data.
These are both specified in Prolog (or ProbLog) files, and they can be combined into one.

The data consists of (probabilistic) facts.
The settings define

* target: the predicate we want to learn
* modes: which predicates can be added to the rules
* types: type information for the predicates
* other settings related to the data

To use:

.. code-block:: bash

    probfoil data.pl

or, in the repository version

.. code-block:: bash

    python probfoil/probfoil.py data.pl

Multiple files can be specified and the information in them is concatenated.
(For example, it is advisable to separate settings from data).

Several command line arguments are available. Use ``--help`` to get more information.

Settings format
---------------

Target
++++++

The target should be specified by adding a fact ``learn(predicate/arity)``.

Modes
+++++

The modes should be specified by adding facts of the form ``mode(predicate(mode1, mode2, ...)``,
where ``modeX`` is the mode specifier for argument X.
Possible mode specifiers are:

   * ``+``: the variable at this position must already exist when the literal is added
   * ``-``: the variable at this position does not exist yet in the rule (note that this is stricter than usual)
   * ``c``: a constant should be introduced here; possible value are derived automatically from the data

Types
+++++

For each relevant predicate (target and modes) there should be a type specifier.
This specifier is of the form ``base(predicate(type1, type2, ...)``, where ``typeX`` is a type identifier.
Type can be identified by arbitrary Prolog atoms (e.g. ``person``, ``a``, etc.)

Example generation
++++++++++++++++++

By default, examples are generated by quering the data for the target predicate.
Negative examples can be specified by adding zero-probability facts, e.g.:

.. code-block:: prolog

    0.0::grandmother(john, mary).

Alternatively, ProbFOIL can derive negative examples automatically by taking combinations of possible
values for the target arguments. Note that this can lead to a combinatorial explosion.
To enable this behavior, you can specify the fact

.. code-block:: prolog

    example_mode(auto).


Example
-------

.. code-block:: prolog

    % Modes
    mode(male(+)).
    mode(parent(+,+)).
    mode(parent(+,-)).
    mode(parent(-,+)).

    % Type definitions
    base(parent(person,person)).
    base(male(person)).
    base(female(person)).
    base(mother(person,person)).
    base(grandmother(person,person)).
    base(father(person,person)).
    base(male_ancestor(person,person)).
    base(female_ancestor(person,person)).

    % Target
    learn(grandmother/2).

    % How to generate negative examples
    example_mode(auto).

Further examples can be found in the directory ``examples``.
