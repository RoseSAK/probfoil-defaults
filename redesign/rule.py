
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
    def __init__(self, parent, literal, previous):
        Rule.__init__(self)
        self.parent = parent
        self.literal = literal
        self.previous = previous

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
        return FOILRuleB(self, literal, self.previous)

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
        head = literals[0]
        body = And.from_list(literals[1:])

        if functor is not None:
            head = Term(functor, *head.args)
        return previous + [Clause(head, body)]

    def __str__(self):
        literals = self.get_literals()
        return '%s :- %s' % (literals[0], ', '.join(map(str, literals[1:])))


class FOILRule(FOILRuleB):
    """Represents the head of a FOILRule.

    :param target: literal in the head of the rule
    :type target: Term
    :param previous: previous rule (if part of set)
    :type previous: FOILRuleB
    """

    def __init__(self, target, previous=None):
        FOILRuleB.__init__(self, None, None, previous)
        self.target = target
        self.previous = previous

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

    def __and__(self, literal):
        """Add a literal to the body of the rule.

        :param literal: literal to add
        :type literal: Term
        :return: new rule
        :rtype: FOILRuleB
        """
        return FOILRuleB(self, literal, self.previous)

    def __str__(self):
        return '%s' % self.target