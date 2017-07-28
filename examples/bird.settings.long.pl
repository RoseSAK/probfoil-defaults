% Modes
mode(bird(+)).
mode(penguin(+)).
mode(eagle(+)).
mode(robin(+)).
mode(blackbird(+)).
mode(dragon(+)).
mode(dog(+)).
mode(ostrich(+)).
%mode(ab_bird(+)).
mode(sparrow(+)).

% Type definitions
base(bird(x)).
base(penguin(x)).
base(flies(x)).
base(blackbird(x)).
base(robin(x)).
base(eagle(x)).
base(dragon(x)).
base(dog(x)).
base(ostrich(x)).
%base(ab_bird(x)).
base(sparrow(x)).

% Target
learn(flies/1).

example_mode(closed).
