% Modes
mode(bird(+)).
mode(penguin(+)).
mode(eagle(+)).
mode(robin(+)).
mode(blackbird(+)).
mode(thrush(+)).
mode(dog(+)).
mode(dodo(+)).
mode(ostrich(+)).
mode(kiwi(+)).

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
base(australian(x)).
base(kiwi(x)).
base(thrush(x)).

% Target
learn(australian/1).

example_mode(closed).

mode(ab_bird(+)).
base(ab_bird(x)).

mode(ab_ab_bird(+)).
base(ab_ab_bird(x)).