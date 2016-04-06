
from problog.logic import Term, Var, Clause, And


class Rule(object):
    """Generic class for rules."""

    def __init__(self):
        pass

    def get_literals(self):
        """Returns all literals in the rule."""
        raise NotImplementedError('abstract method')

    def get_literal(self):
        """Returns the last added literal."""
        raise NotImplementedError('abstract method')

    def get_variables(self):
        """Return the set of variables in the rule.

        :return: all variables in that occur in the rule
        :rtype: set[Var]
        """
        variables = set()
        for lit in self.get_literals():
            variables |= lit.variables()
        return variables

    def to_clauses(self, functor=None):
        """Transform rule into ProbLog clauses

        :param functor: override rule functor (set to None to keep original)
        :type functor: str | None
        :return: clause representation
        :rtype: list[Clause]
        """
        raise NotImplementedError('abstract method')


class FOILRuleB(Rule):
    """A FOIL rule is a rule with a specified target literal.
    These rules are represented as a (reversed) linked list.

    :param parent: parent rule which corresponds to the rule obtained by removing the last literal
    :type parent: FOILRuleB | None
    :param literal: last literal of the body
    :type literal: Term | None
    """
    def __init__(self, parent, literal, target, previous, correct):
        Rule.__init__(self)
        self.parent = parent
        self.literal = literal
        self.target = target
        self.previous = previous
        self.correct = correct
        self.rule_prob = None
        self.score = None

    def get_literals(self):
        """Get literals in the rule.

        :return: list of literals including target
        :rtype: list[Term]
        """
        return self.parent.get_literals() + [self.literal]

    def get_literal(self):
        """Get most recently added body literal.

        :return: None (the body is empty for this type of rule)
        :rtype: Term
        """
        return self.literal

    def __and__(self, literal):
        """Add a literal to the body of the rule.

        :param literal: literal to add
        :type literal: Term
        :return: new rule
        :rtype: FOILRuleB
        """
        return FOILRuleB(self, literal, self.target, self.previous, self.correct)

    def to_clauses(self, functor=None):
        """Transform rule into ProbLog clauses

        :param functor: override rule functor (set to None to keep original)
        :type functor: str | None
        :return: clause representation
        :rtype: list[Clause]
        """
        if self.previous:
            previous = self.previous.to_clauses(functor)
        else:
            previous = []

        literals = self.get_literals()
        head = literals[0].with_probability(self.get_rule_probability())
        body = And.from_list(literals[1:])

        if functor is not None:
            head = Term(functor, *head.args)
        return previous + [Clause(head, body)]

    def set_rule_probability(self, probability=None):
        rule = self
        while rule.parent:
            rule = rule.parent
        rule.rule_prob = probability

    def get_rule_probability(self):
        rule = self
        while rule.parent:
            rule = rule.parent
        return rule.rule_prob

    def __str__(self):
        literals = self.get_literals()
        head = literals[0].with_probability(self.get_rule_probability())
        if len(literals) == 1:
            return '%s :- fail.' % (head,)
        else:
            return '%s :- %s' % (head, ', '.join(map(str, literals[1:])))

    def is_equivalent(self, other):
        if self.score != other.score:
            return False
        literals1 = sorted(self.get_literals())
        literals2 = sorted(self.get_literals())

        return literals1 == literals2



class FOILRule(FOILRuleB):
    """Represents the head of a FOILRule.

    :param target: literal in the head of the rule
    :type target: Term
    :param previous: previous rule (if part of set)
    :type previous: FOILRuleB
    """

    def __init__(self, target=None, previous=None, correct=None):
        if target is None and previous is not None:
            target = previous.target
        FOILRuleB.__init__(self, None, None, target, previous, correct)

    def get_literals(self):
        """Get literals in the rule.

        :return: list of literals including target
        :rtype: list[Term]
        """
        return [self.target]

    def get_literal(self):
        """Get most recently added body literal.

        :return: None (the body is empty for this type of rule)
        :rtype: Term
        """
        return None

    def __str__(self):
        return '%s :- fail' % self.target