

% :- consult('data/myfamily.pl').

% data structure
%
%   level -> [node] 
%   
%   transition -> prev_node, relation
%   
%   for each transition => new node

% ASSUME RELATION ARITY 2

member( H, [H|T]).
member( X, [H|T]) :- member(X,T).

select( H, [H|T], T).
select( X, [H|T], [H|L]) :- select(X, T, L).  

reverse( L, R ) :- reverse( L, [], R ).
reverse( [], R, R).
reverse( [H|T], S, R) :- reverse( T, [H|S], R ).

check_constant( Constant, Level) :-
    ( recorded( Constant, (L,_),_) ->
        L == Level
    ;
        true
    ).
    
node_count( Old, New ) :-
    (nb_getval(node_count, Old) ->
        nb_setval(node_count,New)
    ;
        nb_setval(node_count,New)
    ).
    
success_level( Old, New) :-
    (nb_getval(success_level, Old) ->
        nb_setval(success_level,New)
    ;
        nb_setval(success_level,New)
    ).


store_node( Level, Parent, Relation, ConstantName, Constants, NodeId) :-
    node_count(NodeId, NodeId),
    assertz( node(Level, NodeId, Parent, Relation, ConstantName, Constants) ),
    NextId is NodeId + 1,
    node_count(_, NextId),
    forall( member(C,Constants), 
        ( recorded( C, (L,Nodes),Ref) ->
            erase(Ref),
            recordz(C, (L,[NodeId|Nodes]),_)
        ;
            recordz(C, (Level,[NodeId]),_)
        )
    ).
    
    

init( Constant ) :-
    nb_setval(success_level,0),
    nb_setval(node_count,0),
    %( recorded(K,_,Ref), erase(Ref), fail ; true),
    %retractall(node(_,_,_,_,_,_)),
    store_node( 0, nil, nil, 'V_0', [ Constant ], _).
%    recordz( Constant, (0,[NodeId])).
%    register_constant( Constant, NodeId).

construct_var(Level, VarName) :-
    atomic_list_concat(['V_',Level], VarName).

step( Level, Modes ) :-
    PrevLevel is Level - 1,
    node(PrevLevel, NodeId, _, _, ConstantName, Constants),     % Retrieve a node
    member(Relation, Modes),                                        % Pick one relation mode
    term_variables(Relation, Vars),             
    select( Constant, Vars, [OtherVar]),                                    % Pick one argument and set to constant        
    construct_var(Level, OtherName),
    setof( OtherVar,                                              % Find all constants that can be reached from node existing constants
        Constant^(   member( Constant, Constants),
            call(Relation),
            check_constant(OtherVar, Level)
        ), List),
    List \= [],
    Constant = ConstantName,
    OtherVar = OtherName,                          % Set other variable name.
    store_node( Level, NodeId, Relation, OtherName, List,_),
    success_level(_, Level),
    fail.
step( Level, _ ) :-
    success_level( X, X),
    X == Level.
    
run( Constant, Modes ) :-
    init(Constant), run_from(1, Modes).
    
run_from( N, Modes ) :-
    step(N, Modes), !,
    M is N+1,
    run_from(M, Modes).
run_from(_,_).
    
show_nodes( Level ) :-
    Goal = node( Level, _, _, _, _, _ ),
    call(Goal),
    write(Goal), nl, fail.
    
paths_for_constants( Constants, Paths, MaxLevel ) :-
    setof( (Level,Path), C^( member(C,Constants), paths_for_constant( C, Path,Level, MaxLevel)), Paths).
    
    
    
paths_for_constant( Constant, Path, Level, MaxLevel ) :-
    recorded(Constant, (Level,Nodes), _),
    member(Node,Nodes),
    path_for_node( Node, Path, MaxLevel ).
        
path_for_node(Node, Result, MaxLevel) :-
    node(Level,Node,Parent,Relation,_,_),
    ( ( Parent == nil; MaxLevel == Level) ->
        Result = []
    ;
        Result = [Relation|Path],
        path_for_node( Parent, Path, MaxLevel)
    ).
    
write_path( (Level,List) ) :- construct_var(Level, Var), write(Var), write('|'), write_path(List).
write_path( [H|T] ) :- write( H ), ( T == [] -> true ; write('|'), write_path(T)).  
    

extract_constant( [A,B], Constant, OtherVar) :- 
    (
        var(A) -> OtherVar=A, Constant=B; Constant=A, OtherVar=B
    ).
    
%rpf( Target ) :- findall( Mode, (base(R), R =.. [F|A], F\=TF, length(A,2), length(A2,2), Mode=..[F|A2]), Modes), rpf( Target, -1, Modes, [])
rpf( Target, MaxLevel, Modes, Constants ) :-
    Target =.. [TF|TArgs],
    extract_constant(TArgs, Constant, OtherVar),
    run(Constant, Modes),
    ( Constants == [] ->
        findall( OtherVar, call(Target) , L )
    ;
        L = Constants
    ),
    paths_for_constants( L, Paths, MaxLevel), 
    member(P,Paths), 
    write_path(P),
    nl,
    fail.
rpf( _, _,_,_ ).

main(Data, MaxLevel, Target, Modes, Constants) :-
    consult(Data),
    rpf(Target, MaxLevel, Modes, Constants).

:- unix( argv([Data,MaxLevelS,TargetS,ModesS|Constants]) ), 
    atom_to_term( TargetS, Target, _ ), 
    atom_to_term(ModesS,Modes,_),  
    atom_number( MaxLevelS, MaxLevel), 
    main( Data, MaxLevel, Target, Modes, Constants).