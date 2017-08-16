% Modes
mode(cat(+)).
mode(tabby(+)).
mode(siamese(+)).
%mode(ginger(+)).
mode(injured(+)).
mode(manx(+)).
mode(dog(+)).

% Types
base(cat(x)).
base(tabby(x)).
base(siamese(x)).
%base(ginger(x)).
base(injured(x)).
base(manx(x)).
base(tail(x)).
base(dog(x)).

% Target

learn(tail/1).

example_mode(closed).

mode(ab_cat(+)).
base(ab_cat(x)).