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
mode(cat(+)).
mode(tabby(+)).
mode(siamese(+)).
mode(ginger(+)).
mode(injured(+)).
mode(manx(+)).
mode(tail(+)).

% Type definitions
base(bird(x)).
base(penguin(x)).
base(flies(x)).
base(blackbird(x)).
base(thrush(x)).
base(robin(x)).
base(eagle(x)).
base(dog(x)).
base(dodo(x)).
base(ostrich(x)).
base(cat(x)).
base(tabby(x)).
base(siamese(x)).
base(ginger(x)).
base(injured(x)).
base(manx(x)).
base(tail(x)).

% Target
learn(tail/1).

example_mode(closed).

mode(ab_bird(+)).
base(ab_bird(x)).
