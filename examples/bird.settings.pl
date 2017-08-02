% Modes
mode(bird(+)).
mode(ab_bird(+)).
mode(penguin(+)).
mode(eagle(+)).
mode(robin(+)).
mode(blackbird(+)).
%mode(thrush(+)).
mode(dog(+)).
mode(dodo(+)).
mode(ostrich(+)).


% Type definitions
base(bird(x)).
base(penguin(x)).
base(flies(x)).
base(blackbird(x)).
base(robin(x)).
base(eagle(x)).
base(dog(x)).
base(dodo(x)).
base(ostrich(x)).
base(ab_bird(x)).

% Target
learn(flies/1).

example_mode(closed).
