
from problog.engine import DefaultEngine
from problog.logic import Term
from problog import get_evaluatable


class DataFile(object):
    """Represents a data file. This is a wrapper around a ProbLog file that offers direct
    querying and evaluation functionality.

    :param source: ProbLog logic program
    :type source: LogicProgram
    """

    def __init__(self, *sources):
        self._database = DefaultEngine().prepare(sources[0])
        for source in sources[1:]:
            for clause in source:
                self._database += clause

    def query(self, functor, arity=None, arguments=None):
        """Perform a query on the data.
        Either arity or arguments have to be provided.

        :param functor: functor of the query
        :param arity: number of arguments
        :type arity: int | None
        :param arguments: arguments
        :type arguments: list[Term]
        :return: list of argument tuples
        :rtype: list[tuple[Term]]
        """
        if arguments is None:
            assert arity is not None
            arguments = [None] * arity

        query = Term(functor, *arguments)
        return self._database.engine.query(self._database, query)

    def ground(self, rule, functor=None, arguments=None):
        """Generate ground program for the given rule.

        :param rule: rule to evaluate
        :type rule: Rule
        :param functor: override rule functor
        :type functor: str
        :param arguments: query arguments (None if non-ground)
        :type arguments: list[tuple[Term]] | None
        :return: ground program
        :rtype: LogicFormula
        """
        if rule is None:
            db = self._database
            target = Term(functor)
        else:
            db = self._database.extend()
            target = None
            for clause in rule.to_clauses(functor):
                target = clause.head
                db += clause

        if arguments is not None:
            queries = [target.with_args(*args) for args in arguments]
        else:
            queries = [target]

        return self._database.engine.ground_all(db, queries=queries)

    def evaluate(self, rule, functor=None, arguments=None, ground_program=None):
        """Evaluate the given rule.

        :param rule: rule to evaluate
        :type rule: Rule
        :param functor: override rule functor
        :type functor: str
        :param arguments: query arguments (None if non-ground)
        :type arguments: list[tuple[Term]] | None
        :param ground_program: use pre-existing ground program (perform ground if None)
        :type ground_program: LogicFormula | None
        :return: dictionary of results
        :rtype: dict[Term, float]
        """
        if ground_program is None:
            ground_program = self.ground(rule, functor, arguments)

        knowledge = get_evaluatable().create_from(ground_program)
        return knowledge.evaluate()
