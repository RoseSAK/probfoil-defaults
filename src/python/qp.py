#! /usr/bin/env python

from rule import Literal
from collections import defaultdict

class QueryStorage(object) :
    
    def __init__(self) :
        self.queries = []
        self.querykeys = {}
    
    def getQuery(self, rule, example) :
        key = rule.getQueryKey(example)
        return self.queries[key]
    
    def addQuery(self, rule, example, query) :
        key = self.getQueryKey(rule, example, query) # Find key
        self.rule.setQuery(example, key)
        
    def getQueryKey(self, rule, example, query) :
        query_key = query.canon        
        key = self.querykeys.get(query_key, None)
        if key == None :
            key = len(self.querykeys)
            self.querykeys[query_key] = key
        return key

class Q(object) :
    
    def __init__(self, qp, rule, example) :
        self.pack = qp
        self.clauses = []
        self.substruct = {}
        self.substruct_r = []
        
        self.canon = []
        if rule.previous :
            prev_key = rule.previous.getQueryKey(example)
            self.canon.append( (prev_key, ) )
        
    def getSubStructure(self, struct) :
        l = str(struct)
        if l in self.substruct :
            return self.substruct[l]
        else :
            r = len(self.substruct)
            self.substruct[l] = r
            self.substruct_r.append(struct)
            return r
        
    def addClause(self, content) :
        canon = tuple(self.getSubStructure(c) for c in content)
        self.canon += [ canon ]
        
    def __repr__(self) :
        return str((self.canon, self.substruct_r))
    

class Query(object) :
    
    def __init__(self, args=None, rule=None, parent=None) :
        self.args = args
        self.clauses = []
        self.examples = []
        self.rules = []
        self.parent = parent
        if rule : self.rules.append(rule)
        
    def addClause(self, newclause, subst) :        
        subst1 = dict((x, subst[x] ) for x in subst if not x in self.args )
        subst2 = tuple( (x, subst[x] ) for x in self.args )
        
        self.examples = [subst2]
        newclause1 = [ c.assign(subst1) for c in newclause ]
        
        self.clauses.append(newclause1)
        # TODO add subsumption tests
    
    def _calc_key(self) :
        k = ''
        if self.parent : k = self.parent.key
        return k + ';' + ';'.join( ','.join(map(str,c)) for c in self.clauses )
    
    key = property(_calc_key)
    
    def __iand__(self, other) :
        if self.args == None :
            return other
        else :        
            assert(self.key == other.key)
            self.examples += other.examples
            self.rules += other.rules
            return self
    
    def __repr__(self) :
        return str((self.key, self.examples, self.rules))

class QueryPack(object) :
    
    def __init__(self) :
        pass
    
    
def test() :
    
    q1 = Q(None)
    q1.addClause( [ Literal('a', '1'), Literal('b', '1a') ] )
    q1.addClause( [ Literal('c', '1') ])
    q1.addClause( [ Literal('a', '1', False) ])
    
    q2 = Q(None)
    q2.addClause( [ Literal('a', '2'), Literal('b', '2a') ])    
    q2.addClause( [ Literal('c', '2') ])    
    
    q3 = Q(None)
    q3.addClause( [ Literal('a', '3'), Literal('b', '3b') ])  
    q3.addClause( [ Literal('c', '3') ])    
    
    print (q1, q2, q3)
    
if __name__ == '__main__' :
    test()
        
#         
# # Query for ProbLog
# class Query(object) :
#      
#     def __init__(self, example_id, parent=None) :
#         self.example_id = example_id
#         self.used_facts = defaultdict(set)
#         if parent :
#             self.clauses = parent.clauses[:]
#         else :
#             self.clauses = []
#         
#     def update_used_facts(self, update) :
#         for pred in update :
#             self.used_facts[pred] |= update[pred]
#         
#         
#     key = property( lambda s : str(s) )
#     value = property( lambda s : (s.example_id, s.facts() ) )
#         
#     def __iadd__(self, clause_body) :
#         self.addClause(clause_body)
#         return self
#         
#     def addClause(self, new_clause) :
#         if not new_clause : return
#         new_clause = set(new_clause)
#         i = 0
#         while i < len(self.clauses) :
#             existing_clause = self.clauses[i]
#             if self._test_subsume(existing_clause, new_clause) :
#                 # existing clause subsumes new clause => discard new clause
#                 return
#             elif self._test_subsume(new_clause, existing_clause) :
#                 # new clause subsumes existing clause => discard existing clause
#                 self.clauses.pop(i)
#             else :
#                 i += 1
#         self.clauses.append( new_clause )
#         self.clauses = sorted(self.clauses)
#         
#     def _test_subsume(self, clause1, clause2) :
#         return clause1 <= clause2
#         
#     def __str__(self) :
#         return ';'.join( ','.join( map(str, clause) ) for clause in self.clauses )
#         
#     def facts(self) :
#         result = set([])
#         for clause in self.clauses :
#             result |= clause
#         return result
#         