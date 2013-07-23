
base(card(hand,suit,rank)).
modes(card(+,-,-)).

base(flush(hand)).
learn(flush(hand)).


card(1, h, 2).
card(1, h, 5).
card(1, h, 6).
card(1, h, 10).
card(1, h, ace).
flush(1).

card(2, d, 2).
card(2, h, 5).
card(2, s, 6).
card(2, c, 10).
card(2, s, ace).


%flush(X) :- card(X,S,A), card(X,S,B), card(X,S,C), card(X,S,D), card(X,S,E).	% with distinct variable assignments