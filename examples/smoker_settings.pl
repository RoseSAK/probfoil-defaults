% Modes
% mode(person(+)).
% mode(friend(+, +)).

% Type definitions
base(person(x)).
base(friend(x, x)).
base(smokes(x)).
base(asthma(x)).

% Target
learn(asthma/1).

% Generate negative examples
% example_mode(auto).
