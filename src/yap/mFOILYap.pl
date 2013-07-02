%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mFOIL 
% Saso Dzeroski 
% Jozef Stefan Institute
% 25 August 1991, 20 November 1993
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Quintus prolog declarations

:- unknown(_,fail).

:- dynamic default/3, global_neg/1, current_beam/2.
:- dynamic beam/1, heuristic/1, m/1, significance/1, smallestaccuracy/1.
:- dynamic learn/1, base/1, modes/1, symmetric/2, positive/1, negative/1.
:- multifile learn/1, base/1, modes/1, symmetric/2, positive/1, negative/1.

% End of Quintus prolog declarations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tunable parameters of the algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% don't write out each clause as it is evaluated by default
%talk.

% write out the beam after each refinement step by default
writebeam.

% beam width of the beam search
beam(5).

% heuristic for guiding the search
heuristic(laplace).%laplace/m

% significance threshold: 99%
significance(6.64).%6.64

% a clause generated has to cover at least 80% of positive examples
smallestaccuracy(0.8).%0.8

% value of the m-parameter in the m-estimate
m(10).

% literals of the form p(X,X) or q(A,X,Y,X) are not allowed
% arguments of literals have to be distinct variables
variables(different).%different/identical

% set(+,+) - set Parameter to Value
set(Parameter,Value) :-
  ParFact =.. [Parameter,_V],
  ( retract(ParFact), ! ; true),
  NewPar =.. [Parameter,Value],
  assertz(NewPar).

%%%%%%%%%%%%%%%%%%%%%
% top level algorithm
%%%%%%%%%%%%%%%%%%%%%

run(InFile,OutFile) :-
  consult(InFile),
  run,
  save_learned(OutFile),
  cleanup.

run :-
  learn(Head),
  snips( ( prepare_for_learning(Head),
           foil(Head),
           local_cleanup ) ),
  fail.
run. 

save_learned(Outfile) :-
  tell(Outfile),
  saveall,
  told.

saveall :-
  %listing(learned),
  save_clauses.

local_cleanup :- 
  retract(global_neg(_)),
  retractall(default(_,_,_)),
  retract(current_beam(_,_)).

save_clauses :-
  learned(H,B,_,_,_),
  writevars((H :- B)),write('.'),
  nl,
  fail.

save_clauses.

cleanup :-
  retractall(positive(_)), 
  retractall(negative(_)),
  retractall(learn(_)),
  retractall(learned(_,_,_,_,_)),
  (retract(base(_X=_Y)), !; true),
  cleanbk,
  retractall(modes(_)),
  retractall(symmetric(_,_)).

cleanbk :- 
  retract(base(P)), 
  new_variables(P,H,_),
  retractall(H),
  fail.
cleanbk.

prepare_for_learning(Head) :-
  calculate_default_accuracy(Head).

calculate_default_accuracy(Head) :-
  new_variables(Head,VH,_),
  VH =.. [_Rel|Args],
  ebagof(Args,positive(VH),Pos),
  ebagof(Args,negative(VH),Neg),
  assertz(global_neg(Neg)),
  length(Pos,Tplus),
  length(Neg,Tminus),
  (Tplus=0, Tminus=0, !, write('No examples for '), writevars(Head), nl ; true),
  DP is Tplus / ( Tplus + Tminus ),
  DN is Tminus / ( Tplus + Tminus ),
  assertz(default(VH,+,DP)),
  assertz(default(VH,-,DN)).
    
  
foil(Head) :-
 new_variables(Head,VarHead,HeadVars),
 write('Learning predicate '), writevars(VarHead), nl, nl,
 repeat,
  initialize(VarHead,HeadVars,AllPos,_AllNeg,_DefValue),
  (
    \+ possibly_significant(AllPos), !
  ;
    snips(find_clause(VarHead, clause(Body : Vars, Value, Pos, Neg))),
    (
      stopping_criterion(Pos,Neg,Value), !
    ;
      snips((
        save_clause(VarHead,clause(Body : Vars, Value, Pos, Neg)),
        remove_positive_examples(VarHead,Pos)
      )),
      fail
    )
  ).


stopping_criterion(Pos,Neg,_Value) :-
  \+ significant(Pos,Neg).

stopping_criterion(Pos,Neg,_) :-
  smallestaccuracy(Small),
  length(Pos,P), length(Neg,N),
  Majority is P/(P+N),
  Majority < Small.


possibly_significant(Pos) :-
  length(Pos,Tplus),
  default(_,+,P1),
  significance(Threshold),
  log0(P1,LogP1),
  2*Tplus*(- LogP1) > Threshold.

significant(Pos,Neg) :-
  length(Pos,Tplus),
  length(Neg,Tminus),
  N is Tplus + Tminus,
  Q1 is Tplus/N, Q2 is Tminus/N,
  default(_,+,P1), default(_,-,P2),
  R1 is Q1/P1, R2 is Q2/P2,
  log0(R1,LR1), log0(R2,LR2),
  ChiSq is 2*N*(Q1*LR1 + Q2*LR2),
  significance(Threshold),
  ChiSq > Threshold.

log0(X,LogX) :-
  ( X =< 0, !, LogX = 0 ; LogX is log(X) ).


initialize(Head,HeadVars,Pos,Neg,Value) :-
  Head =.. [_Rel|Args],
  ebagof(Args,positive(Head),Pos),
  global_neg(Neg),
  length(Pos,Tplus),
  length(Neg,Tminus),
  h_evaluate(Tplus,Tminus,Value),
  write('H_EVAL: '), writeln(Value),
  assertz(current_beam(Head,[clause(true : HeadVars, Value, Pos, Neg)])).


% find_clause( +, -)
% find_clause(Head,Clause)

find_clause(Head,Clause) :-
  repeat,
    retract(current_beam(Head,CurrentBodies)),
    best_refinements(Head,CurrentBodies,[],Bodies),
    reverse(CurrentBodies,RCB),
    sorted_filter(Bodies,RCB,TmpBodies),
    reverse(TmpBodies,Sorted),
    snips((writebeam, write('Current beam: '), nl, writebeam(Head,Sorted); true)),
    CurrentBodies = [clause(BC,VC,CP,NC)|_R], Sorted = [clause(BT,VT,TP,NT)|_S], 
    (
      \+ worse(clause(BC,VC,CP,NC),clause(BT,VT,TP,NT)),
      !,
      Clause = clause(BC,VC,CP,NC)
    ;
      assertz(current_beam(Head,Sorted)),
      fail
    ).

writebeam(H,S) :-
  freshcopy((H,S),(NH,NS)),
  numbervars((NH,NS),0,_NVARS),
  write(NH), write(':-'), nl, 
  write_beam(NS).
write_beam([]).
write_beam([clause(Body,Value,P,N)|R]) :-
  write(Body), write(' '),
  write(Value), write(' '),
  length(P,LP),length(N,LN), write(': '),write(LP),write(' '),write(LN),nl,
  write_beam(R).

% sorted_filter(+NewBest, +CurrentBest, -Best)

sorted_filter([],X,X).
sorted_filter([C|R],L,N) :-
  update(C,L,T),
  sorted_filter(R,T,N).

% best_refinements(+Head,+Bodies,+CurrentBest,-Best)
% find the list Best of best refinements of Bodies
% CurrentBest and Best are of length at most BeamWidth

best_refinements(_Head,[],Best,Best).
best_refinements(Head,[clause(Body,Val,Pos,Neg)|Bodies],CurrentBest,Best):-
  refinements(Head,Body,LiteralList),
  evaluate_list(Head,Body,Val,Pos,Neg,LiteralList,CurrentBest,NewBest),
  best_refinements(Head,Bodies,NewBest,Best).


% refinements( +, +, -)
% find all possible literals that can be added to the body of the clause

refinements(Head,Body,LiteralList) :-
  ebagof(Predicate:Literals, (base(Predicate), literals(Head,Body,Predicate, Literals)), LiteralListOfLists),
  collect(LiteralListOfLists,LiteralList).

collect([],[]).
collect([_Predicate:Literals|R], LList) :-
  collect(R,LL1),
  append(Literals,LL1,LList).


% literals(+, +, +, -)
% find all literals of type Predicate that can be added to the clause

literals(Head,Body:OldVars,Predicate,LiteralList) :-
  identical(Head,Predicate), !,
  bodytolist(Body,CondList),
  new_variables(Predicate,VarPred,VarList),
  freshcopy(VarPred,ModePred), ModePred=.. [Name|Modes],
  choose(VarList,OldVars,[],Modes,CList),
  Predicate=..[Name|Types],
  Head=..[Name|HeadArgs],
  pairs(HeadArgs,Types,HeadList),
  del_one(HeadList,CList,ChoiceList,[]),
  ebagof(VarPred:VarList,
	 (member(VarList,ChoiceList), \+ imember(VarPred,CondList)),
	 LiteralList).

literals(_Head,Body:OldVars,Predicate,LiteralList) :-
  identical((_X=_Y),Predicate), !,
  bodytolist(Body,CondList),
  ebagof((A=B):[A:Type,B:Type],
         (member(A:Type,OldVars), member(B:Type,OldVars), 
	 \+ imember((A=B),CondList), \+ imember((B=A),CondList), A \== B),
	 LL),
  del_equalities(LL,LiteralList).
  
literals(_Head,Body:Vars,attribute_value(T),LiteralList) :-
  !,
  bodytolist(Body,CondList),
  type(T,Values),
  ebagof( (X=V):[X:T], 
          ( member(X:T,Vars), 
	    \+ ( member(C,Values), imember((X=C),CondList) ),
	    member(V,Values) ), 
          LiteralList ).

literals(_Head,Body:OldVars,Predicate,LiteralList) :-
  bodytolist(Body,CondList),
  new_variables(Predicate, VarPred,VarList),
  new_variables(Predicate,_,NewVars),
  freshcopy(VarPred,ModePred), modes(ModePred), ModePred=.. [_Name|Modes],
  choose(VarList,OldVars,NewVars,Modes,ChList),
  ( variables(different),!, 
    del_same_vars(ChList,Ch1List)
   ;
    Ch1List == Chlist),
  remove_symmetries(VarPred,Ch1List,ChoiceList),
  ebagof(VarPred:VarList,
	 (member(VarList,ChoiceList), \+ imember(VarPred,CondList)),
	 LiteralList).


bodytolist(true,[]).
bodytolist((B,L),[L|R]) :- 
  bodytolist(B,R).

del_same_vars([],[]).
del_same_vars([S|R],T) :-
  member(X,S),
  snips( ( del_one(X,S,S1,[]),
           imember(X,S1) ) ),
  !,
  del_same_vars(R,T).
del_same_vars([S|R],[S|T]) :-
  del_same_vars(R,T).


% remove_symmetries(+Predicate,+ListOfArgumentChoices,-ClearedList)

remove_symmetries(Head,ChList,CleanList) :-
  symmetries(Head,SymmetryList),
  del_symmetries(SymmetryList,ChList,CleanList).

symmetries(H,SL) :- 
  symmetric(H,SL), 
  !.
symmetries(_,[]).

del_symmetries([],C,C).
del_symmetries([(X,Y)|R],C,NC) :-
  del_symmetry((X,Y),C,NC1),
  del_symmetries(R,NC1,NC).

del_symmetry(_,[],[]).
del_symmetry((X,Y), [Choice|List], [Choice|NCL]) :-
  swap(X,Y,Choice,NChoice),
  del_one(NChoice,List,L1,[]),
  del_symmetry((X,Y),L1,NCL).

% swap( +FirstPosition, +SecondPosition, +ArgList, -SwappedArgList)
swap(X,Y,C,NC) :-
  del_pos(X,C,D1,V1,D2),
  Z is Y-X,
  del_pos(Z,D2,D3,V2,D4),
  append(D3,[V1|D4],D5),
  append(D1,[V2|D5],NC).

del_pos(1,[X|R],[],X,R).
del_pos(N,[X|R],[X|L],Y,T) :-
  N > 1,
  N1 is N - 1,
  del_pos(N1,R,L,Y,T).


identical(Head,Predicate) :-
  freshcopy(Head,H),
  freshcopy(Predicate,P),
  H = P.

del_equalities([],[]).
del_equalities([(X=Y):[X:T,Y:T]|R],[(X=Y):[X:T,Y:T]|D]) :-
  del_one((Y=X):[Y:T,X:T],R,S,[]),
  del_equalities(S,D).


% new_variables( +, +, -)
% example: new_variables(adj(file,file),adj(X,Y),[X:file,Y:file])
% create a list of new variable, variabilize Head with them
% to obtain VarHead and associate variables with corresponding types  HeadVars

new_variables(Head,VarHead,HeadVars) :-
  Head =.. [Name|Types],
  length(Types,Arity),
  varlist(Arity,Vars),
  VarHead =.. [Name|Vars],
  pairs(Vars,Types,HeadVars).


% choose( +, +, +, +, -)
% create all possible variable lists (VarList) from given old and new variables
% take into account type constraints, i.e., types of variables and mode decls.

choose(Args,OldVars,NewVars,Modes,ChoiceList) :-
  append(OldVars,NewVars,Vars),
  ebagof(Args, variabilize(Args,OldVars,Vars,Modes), VarList),
  del_duplicates(VarList,ChoiceList,NewVars).

% variabilize( -, +, +)
% choose Args from given variables so that at least one variable is old

variabilize(Args, OldVars, Vars, Modes) :-
  delete(X, Args, A1, A2), 
  length(A1,N1), 
  N is N1 + 1,
  del_pos(N,Modes,M1,_,M2),
  member(X, OldVars),
  lsubset(A1, Vars, OldVars, M1),
  lsubset(A2, Vars, OldVars, M2).

lsubset([],_,_,_).
lsubset([H|T],All,Old,[+|R]) :-
  !,
  member(H,Old),
  lsubset(T,All,Old,R).
lsubset([H|T],All,Old,[-|R]) :-
  member(H,All),
  lsubset(T,All,Old,R).


% evaluate_list(+Head,+Body,+Val,+Pos,+Neg,+LiteralList,+CurrentBest,-NewBest)
% Pos and Neg are lists of examples covered by Body
% LiteralList contains possible additions to Body
% CurrentBest is a list of length at most BeamWidth

evaluate_list(_Head,_Body:_Vars,  _,_Pos,_Neg,[],Best,Best).
evaluate_list(Head,Body:Vars,Val,Pos,Neg,[L:LVars|Ls],CurrBest,NewBest) :-
  evaluate(Head,Body,L,Pos,Neg,PPos,PNeg,PHValue,NPos,NNeg,NHValue),
  update_vars(Vars,LVars,NewVars),
 snips(
   (PHValue > Val, 
    update(clause((Body, L) : NewVars,PHValue,PPos,PNeg),CurrBest, TempBest1)
    ;
    TempBest1 = CurrBest ) ),
  snips(
   (NHValue > Val,
    update(clause((Body,\+ L) : Vars, NHValue,NPos,NNeg),TempBest1,TempBest2)
    ;
    TempBest2 = TempBest1 ) ),
  evaluate_list(Head,Body:Vars,Val,Pos,Neg,Ls,TempBest2,NewBest).


% evaluate(+Head,+Body,+L,+Pos,+Neg,-PPos,-PNeg,-PHV,-NPos,-NNeg,-NHV)
% given Body which covers Pos and Neg examples, compute the sets 
% PPos and PNeg (covered by Body&L), NPos and NNeg (covered by Body&~L)
% as well as the heuristics' values PHV and NHV of (Body,L) and (Body, not L)

evaluate(Head,Body,L,Pos,Neg,PPos,PNeg,PHV,NPos,NNeg,NHV) :-
  covers(Head,(Body,L),Pos,[],[],PPos,NPos),
  covers(Head,(Body,L),Neg,[],[],PNeg,NNeg),

  length(PPos,Tplus),
  length(PNeg,Tminus),
  length(NPos,Nplus),
  length(NNeg,Nminus),
  
  h_evaluate(Tplus,Tminus,PHV),
  h_evaluate(Nplus,Nminus,NHV),

 snips((talk, writevars((Head :- (Body, L))), nl, 
        write(' '), write(Tplus), write(' '), 
        write(Tminus), write(' '), write(PHV),
        write('|'), write(Nplus), write(' '), 
        write(Nminus), write(' '), write(NHV),
        nl; true)).

% h_evaluate(+NoOfPosExCvd, +NoOfNegExCvd, -HeuristicValue)

h_evaluate(Pos,Neg,Val) :-
  heuristic(laplace),
  laplace(Pos,Neg,Val).
h_evaluate(Pos,Neg,Val) :-
  heuristic(m),
  m_estimate(Pos,Neg,Val).


laplace(TP,FP,Lap) :-
  ( TP = 0, !, Lap = 0
   ;
    Lap is ( TP + 1 ) / ( TP + FP + 2 )).

m_estimate(TP,FP,Mest) :-
  m(M),
  ( TP = 0,  M=0, !, Mest = 0
   ;
    default(_,+,DP),
    Mest is ( TP + M*DP ) / ( TP + FP + M )).


% update_vars(+OldVars,+LitVars,-NewVars) 
% if there are new variables in LitVars, add them to OldVars to get  NewVars

update_vars(L,[],L).
update_vars(L,[X|R],S) :-
  imember(X,L), !,
  update_vars(L,R,S).
update_vars(L,[X|R],S) :-
  append(L,[X],L1),
  update_vars(L1,R,S).


% update(+clause(Body,Gain,Pos,Neg), +CurrBest, -NewBest)
% if Gain is high enough or CurrentBest is empty enough add Body
% to best bodies found so far

update(clause(Body,Value,Pos,Neg), B, N) :-
  Value > 0,
  length(B,LB), beam(BW), LB < BW,
  possibly_significant(Pos),
  !,
  insert_sorted(clause(Body,Value,Pos,Neg),B,N).
update(clause(Body,Value,Pos,Neg), [clause(B,V1,P,N)|Bodies], NewBodies) :-
  snips(worse(clause(B,V1,P,N),clause(Body,Value,Pos,Neg))),
  possibly_significant(Pos),
  \+ already_in(clause(Body,Value,Pos,Neg),Bodies), 
  !,
  insert_sorted(clause(Body,Value,Pos,Neg),Bodies,NewBodies).
update(_, Bodies, Bodies).


already_in(clause(B:_Vs,V,P,N),BS) :-
  member(clause(NB:_NVs,V,P,N),BS),
  bodytolist(B,BL), bodytolist(NB,NL),
  subset(NL,BL).

subset([],_).
subset([X|R],L) :-
  imember(X,L),
  subset(R,L).

insert_sorted(X,[],[X]).
insert_sorted(X,[Y|T],[X,Y|T]) :-
  worse(X,Y), 
  !.
insert_sorted(X,[Y|T],[Y|U]) :-
  insert_sorted(X,T,U).

	   
worse(clause(BX,GX,PX,NX),clause(BY,GY,PY,NY)) :-
  GX < GY, !.
worse(clause(BX:VX,GX,PX,NX),clause(BY:VY,GY,PY,NY)) :-
  GX = GY, 
  length(PX,LX), length(PY,LY), 
  ( LX < LY, ! ; 
    bodylength(BX,BLX), bodylength(BY,BLY), BLX > BLY ).

bodylength(true,0).
bodylength((B,_L),N) :-
  bodylength(B,N1),
  N is N1 + 1.


covers(_,_,[],PSet,NSet,PSet,NSet).
covers(Head,Body,[X|Set],P,N,PSet,NSet) :-
  ( covers(Head,Body,X),
    !,
    covers(Head,Body,Set,[X|P],N,PSet,NSet)
  ;
    covers(Head,Body,Set,P,[X|N],PSet,NSet)).

covers(Head,Body,Example) :-
  freshcopy((Head,Body),(H1,B1)),
  H1 =.. [_Rel|Example],
  writeln(B1),
  call(B1), !.


positive_uncovered(Head) :-
  Head =.. [Name|Args],
  length(Args,Arity),
  varlist(Arity,Vars),
  NH =.. [Name|Vars],
  positive(NH).


better_than_default(Head,Accuracy) :-
  default(Head,+,Default),
  Accuracy > Default.


save_clause(VarHead,clause(Body : _Vars,     Value,   Pos,   Neg)) :-
  length(Pos,NPos),
  length(Neg,NNeg),
  assertz(learned(VarHead,Body,Value,NPos,NNeg)),
  write('Clause generated: '), writevars((VarHead :- Body)),
  nl.


remove_positive_examples(Head,Pos) :-
  Head =.. [_Rel|X],
  member(X,Pos),
  snips(retract(positive(Head))),
  fail.

remove_positive_examples(_,_).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Various list processing utilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

member(X,[X|_]).
member(X,[_|Y]) :- member(X,Y).

append([],L,L).
append([X|T],S,[X|A]) :- 
  append(T,S,A).

reverse([],[]).
reverse([X|L],R) :-
  reverse(L,Lr),
  append(Lr,[X],R).

imember(X,[Y|_]) :- X==Y.
imember(X,[_|Y]) :- imember(X,Y).


% variant(+X, +Y, +NewVariables)
% given a predicate symbol p and a clause c with new variables NewVars 
% let L1=[p|X] and L2=[p|Y]. check if adding each of L1 and L2 to c will
% produce clauses that are alphabetic variants of each other

variant(X,Y,_) :- X==Y, !.
variant([A|R],[B|S],NewVars) :-
  (A == B, !;
   imember(A,NewVars),
   imember(B,NewVars)),
  variant(R,S,NewVars).

vmember(X,[Y|_],NV) :-
  variant(X,Y,NV).
vmember(X,[_|Y],NV) :-
  vmember(X,Y,NV).

% varlist(+Length,-VariableList)
% create a list of new variables of length Length

varlist(0,[]) :- !.
varlist(Arity, Args) :-
  member1(_, Args, Arity).

member1(X,[X],1).
member1(X,[_|T],N) :-
   N>0,
   N1 is N-1,
   member1(X,T,N1).

pairs([],[],[]).
pairs([X|R],[Y|S],[X:Y|T]) :-
  pairs(R,S,T).

delete(X, [X|A], [], A).
delete(X, [Y|A], [Y|B], C) :-
  delete(X, A, B, C).


% del_duplicates(+ListOfArgumentLists,-NewListOfArgumentLists,+NewVariables)
% delete lists of arguments that would produce alphabetic variants when 
% added to the current clause, given ne w variables NewVariables

del_duplicates([],[],_).
del_duplicates([H|T],[H|S],NV) :-
  del_all(H,T,L,NV),
  del_duplicates(L,S,NV).

del_all(X,L,LX,NV) :-
  vmember(X,L,NV),
  !,
  del_one(X,L,LY,NV),
  del_all(X,LY,LX,NV).
del_all(_X,L,L,_NV).

del_one(_,[],[],_).
del_one(X,[Y|L],L,NV) :-
  variant(X,Y,NV), !.
del_one(X,[Y|L],[Y|LD],NV) :-
  del_one(X,L,LD,NV).


%%%%%%%%%%%%%%%%%%%%%%%%%
% Miscellaneous utilities
%%%%%%%%%%%%%%%%%%%%%%%%%

% snips(+X)
% prevent backtracking over X. same as [! X !] in Arity prolog
snips(X) :-
  call(X), !.

% freshcopy(+X,-Y)
% get a copy of term X with fresh variables
freshcopy(X,Y) :-
  assertz(newcopy(X)),
  retract(newcopy(Y)),
  !.

% no solutions means an empty list of solutions and not a failure

ebagof(X,Y,Z) :-
	bagof(X,Y,Z),
	!.
ebagof(_X,_Y,[]).

esetof(X,Y,Z) :- setof(X,Y,Z), !.
esetof(_X,_Y,[]).

% write terms with variables nicely
writevars(X) :-
  freshcopy(X,Y),
  numbervars(Y,0,_L),
  write(Y).
