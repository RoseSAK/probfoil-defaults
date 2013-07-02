
% base(parent(person,person)).
% modes(parent(-,-)).
% base(male(person)).
% modes(male(+)). 
% base(female(person)).
% modes(female(+)).
% 
% learn(mother(person,person)).
% modes(mother(-,-)).
% 
% learn(father(person,person)).
% modes(father(-,-)).
% 
% learn(male_ancestor(person,person)).
% base(male_ancestor(person,person)).
% modes(male_ancestor(-,-)).
% 
% learn(female_ancestor(person,person)).
% base(female_ancestor(person,person)).
% modes(female_ancestor(-,-)).


% positive(mammal(X)) :- animal(X), mammal(X).
% negative(mammal(X)) :- animal(X), \+mammal(X).

animal(aardvark).
has_hair(aardvark).
gives_milk(aardvark).
is_predator(aardvark).
is_toothed(aardvark).
has_backbone(aardvark).
breathes(aardvark).
has_legs(aardvark,4).
is_catsize(aardvark).
mammal(aardvark).

animal(antelope).
has_hair(antelope).
gives_milk(antelope).
is_toothed(antelope).
has_backbone(antelope).
breathes(antelope).
has_legs(antelope,4).
has_tail(antelope).
is_catsize(antelope).
mammal(antelope).

animal(bass).
lays_eggs(bass).
is_aquatic(bass).
is_predator(bass).
is_toothed(bass).
has_backbone(bass).
has_fins(bass).
has_tail(bass).
fish(bass).

animal(bear).
has_hair(bear).
gives_milk(bear).
is_predator(bear).
is_toothed(bear).
has_backbone(bear).
breathes(bear).
has_legs(bear,4).
is_catsize(bear).
mammal(bear).

animal(boar).
has_hair(boar).
gives_milk(boar).
is_predator(boar).
is_toothed(boar).
has_backbone(boar).
breathes(boar).
has_legs(boar,4).
has_tail(boar).
is_catsize(boar).
mammal(boar).

animal(buffalo).
has_hair(buffalo).
gives_milk(buffalo).
is_toothed(buffalo).
has_backbone(buffalo).
breathes(buffalo).
has_legs(buffalo,4).
has_tail(buffalo).
is_catsize(buffalo).
mammal(buffalo).

animal(calf).
has_hair(calf).
gives_milk(calf).
is_toothed(calf).
has_backbone(calf).
breathes(calf).
has_legs(calf,4).
has_tail(calf).
is_domestic(calf).
is_catsize(calf).
mammal(calf).

animal(carp).
lays_eggs(carp).
is_aquatic(carp).
is_toothed(carp).
has_backbone(carp).
has_fins(carp).
has_tail(carp).
is_domestic(carp).
fish(carp).

animal(catfish).
lays_eggs(catfish).
is_aquatic(catfish).
is_predator(catfish).
is_toothed(catfish).
has_backbone(catfish).
has_fins(catfish).
has_tail(catfish).
fish(catfish).

animal(cavy).
has_hair(cavy).
gives_milk(cavy).
is_toothed(cavy).
has_backbone(cavy).
breathes(cavy).
has_legs(cavy,4).
is_domestic(cavy).
mammal(cavy).

animal(cheetah).
has_hair(cheetah).
gives_milk(cheetah).
is_predator(cheetah).
is_toothed(cheetah).
has_backbone(cheetah).
breathes(cheetah).
has_legs(cheetah,4).
has_tail(cheetah).
is_catsize(cheetah).
mammal(cheetah).

animal(chicken).
has_feathers(chicken).
lays_eggs(chicken).
is_airborne(chicken).
has_backbone(chicken).
breathes(chicken).
has_legs(chicken,2).
has_tail(chicken).
is_domestic(chicken).
bird(chicken).

animal(chub).
lays_eggs(chub).
is_aquatic(chub).
is_predator(chub).
is_toothed(chub).
has_backbone(chub).
has_fins(chub).
has_tail(chub).
fish(chub).

animal(clam).
lays_eggs(clam).
is_predator(clam).
invertebrate(clam).

animal(crab).
lays_eggs(crab).
is_aquatic(crab).
is_predator(crab).
has_legs(crab,4).
invertebrate(crab).

animal(crayfish).
lays_eggs(crayfish).
is_aquatic(crayfish).
is_predator(crayfish).
has_legs(crayfish,6).
invertebrate(crayfish).

animal(crow).
has_feathers(crow).
lays_eggs(crow).
is_airborne(crow).
is_predator(crow).
has_backbone(crow).
breathes(crow).
has_legs(crow,2).
has_tail(crow).
bird(crow).

animal(deer).
has_hair(deer).
gives_milk(deer).
is_toothed(deer).
has_backbone(deer).
breathes(deer).
has_legs(deer,4).
has_tail(deer).
is_catsize(deer).
mammal(deer).

animal(dogfish).
lays_eggs(dogfish).
is_aquatic(dogfish).
is_predator(dogfish).
is_toothed(dogfish).
has_backbone(dogfish).
has_fins(dogfish).
has_tail(dogfish).
is_catsize(dogfish).
fish(dogfish).

animal(dolphin).
gives_milk(dolphin).
is_aquatic(dolphin).
is_predator(dolphin).
is_toothed(dolphin).
has_backbone(dolphin).
breathes(dolphin).
has_fins(dolphin).
has_tail(dolphin).
is_catsize(dolphin).
mammal(dolphin).

animal(dove).
has_feathers(dove).
lays_eggs(dove).
is_airborne(dove).
has_backbone(dove).
breathes(dove).
has_legs(dove,2).
has_tail(dove).
is_domestic(dove).
bird(dove).

animal(duck).
has_feathers(duck).
lays_eggs(duck).
is_airborne(duck).
is_aquatic(duck).
has_backbone(duck).
breathes(duck).
has_legs(duck,2).
has_tail(duck).
bird(duck).

animal(elephant).
has_hair(elephant).
gives_milk(elephant).
is_toothed(elephant).
has_backbone(elephant).
breathes(elephant).
has_legs(elephant,4).
has_tail(elephant).
is_catsize(elephant).
mammal(elephant).

animal(flamingo).
has_feathers(flamingo).
lays_eggs(flamingo).
is_airborne(flamingo).
has_backbone(flamingo).
breathes(flamingo).
has_legs(flamingo,2).
has_tail(flamingo).
is_catsize(flamingo).
bird(flamingo).

animal(flea).
lays_eggs(flea).
breathes(flea).
has_legs(flea,6).
insect(flea).

animal(frog).
lays_eggs(frog).
is_aquatic(frog).
is_predator(frog).
is_toothed(frog).
has_backbone(frog).
breathes(frog).
has_legs(frog,4).
amphibian(frog).

animal(frog).
lays_eggs(frog).
is_aquatic(frog).
is_predator(frog).
is_toothed(frog).
has_backbone(frog).
breathes(frog).
is_venomous(frog).
has_legs(frog,4).
amphibian(frog).

animal(fruitbat).
has_hair(fruitbat).
gives_milk(fruitbat).
is_airborne(fruitbat).
is_toothed(fruitbat).
has_backbone(fruitbat).
breathes(fruitbat).
has_legs(fruitbat,2).
has_tail(fruitbat).
mammal(fruitbat).

animal(giraffe).
has_hair(giraffe).
gives_milk(giraffe).
is_toothed(giraffe).
has_backbone(giraffe).
breathes(giraffe).
has_legs(giraffe,4).
has_tail(giraffe).
is_catsize(giraffe).
mammal(giraffe).

animal(girl).
has_hair(girl).
gives_milk(girl).
is_predator(girl).
is_toothed(girl).
has_backbone(girl).
breathes(girl).
has_legs(girl,2).
is_domestic(girl).
is_catsize(girl).
mammal(girl).

animal(gnat).
lays_eggs(gnat).
is_airborne(gnat).
breathes(gnat).
has_legs(gnat,6).
insect(gnat).

animal(goat).
has_hair(goat).
gives_milk(goat).
is_toothed(goat).
has_backbone(goat).
breathes(goat).
has_legs(goat,4).
has_tail(goat).
is_domestic(goat).
is_catsize(goat).
mammal(goat).

animal(gorilla).
has_hair(gorilla).
gives_milk(gorilla).
is_toothed(gorilla).
has_backbone(gorilla).
breathes(gorilla).
has_legs(gorilla,2).
is_catsize(gorilla).
mammal(gorilla).

animal(gull).
has_feathers(gull).
lays_eggs(gull).
is_airborne(gull).
is_aquatic(gull).
is_predator(gull).
has_backbone(gull).
breathes(gull).
has_legs(gull,2).
has_tail(gull).
bird(gull).

animal(haddock).
lays_eggs(haddock).
is_aquatic(haddock).
is_toothed(haddock).
has_backbone(haddock).
has_fins(haddock).
has_tail(haddock).
fish(haddock).

animal(hamster).
has_hair(hamster).
gives_milk(hamster).
is_toothed(hamster).
has_backbone(hamster).
breathes(hamster).
has_legs(hamster,4).
has_tail(hamster).
is_domestic(hamster).
mammal(hamster).

animal(hare).
has_hair(hare).
gives_milk(hare).
is_toothed(hare).
has_backbone(hare).
breathes(hare).
has_legs(hare,4).
has_tail(hare).
mammal(hare).

animal(hawk).
has_feathers(hawk).
lays_eggs(hawk).
is_airborne(hawk).
is_predator(hawk).
has_backbone(hawk).
breathes(hawk).
has_legs(hawk,2).
has_tail(hawk).
bird(hawk).

animal(herring).
lays_eggs(herring).
is_aquatic(herring).
is_predator(herring).
is_toothed(herring).
has_backbone(herring).
has_fins(herring).
has_tail(herring).
fish(herring).

animal(honeybee).
has_hair(honeybee).
lays_eggs(honeybee).
is_airborne(honeybee).
breathes(honeybee).
is_venomous(honeybee).
has_legs(honeybee,6).
is_domestic(honeybee).
insect(honeybee).

animal(housefly).
has_hair(housefly).
lays_eggs(housefly).
is_airborne(housefly).
breathes(housefly).
has_legs(housefly,6).
insect(housefly).

animal(kiwi).
has_feathers(kiwi).
lays_eggs(kiwi).
is_predator(kiwi).
has_backbone(kiwi).
breathes(kiwi).
has_legs(kiwi,2).
has_tail(kiwi).
bird(kiwi).

animal(ladybird).
lays_eggs(ladybird).
is_airborne(ladybird).
is_predator(ladybird).
breathes(ladybird).
has_legs(ladybird,6).
insect(ladybird).

animal(lark).
has_feathers(lark).
lays_eggs(lark).
is_airborne(lark).
has_backbone(lark).
breathes(lark).
has_legs(lark,2).
has_tail(lark).
bird(lark).

animal(leopard).
has_hair(leopard).
gives_milk(leopard).
is_predator(leopard).
is_toothed(leopard).
has_backbone(leopard).
breathes(leopard).
has_legs(leopard,4).
has_tail(leopard).
is_catsize(leopard).
mammal(leopard).

animal(lion).
has_hair(lion).
gives_milk(lion).
is_predator(lion).
is_toothed(lion).
has_backbone(lion).
breathes(lion).
has_legs(lion,4).
has_tail(lion).
is_catsize(lion).
mammal(lion).

animal(lobster).
lays_eggs(lobster).
is_aquatic(lobster).
is_predator(lobster).
has_legs(lobster,6).
invertebrate(lobster).

animal(lynx).
has_hair(lynx).
gives_milk(lynx).
is_predator(lynx).
is_toothed(lynx).
has_backbone(lynx).
breathes(lynx).
has_legs(lynx,4).
has_tail(lynx).
is_catsize(lynx).
mammal(lynx).

animal(mink).
has_hair(mink).
gives_milk(mink).
is_aquatic(mink).
is_predator(mink).
is_toothed(mink).
has_backbone(mink).
breathes(mink).
has_legs(mink,4).
has_tail(mink).
is_catsize(mink).
mammal(mink).

animal(mole).
has_hair(mole).
gives_milk(mole).
is_predator(mole).
is_toothed(mole).
has_backbone(mole).
breathes(mole).
has_legs(mole,4).
has_tail(mole).
mammal(mole).

animal(mongoose).
has_hair(mongoose).
gives_milk(mongoose).
is_predator(mongoose).
is_toothed(mongoose).
has_backbone(mongoose).
breathes(mongoose).
has_legs(mongoose,4).
has_tail(mongoose).
is_catsize(mongoose).
mammal(mongoose).

animal(moth).
has_hair(moth).
lays_eggs(moth).
is_airborne(moth).
breathes(moth).
has_legs(moth,6).
insect(moth).

animal(newt).
lays_eggs(newt).
is_aquatic(newt).
is_predator(newt).
is_toothed(newt).
has_backbone(newt).
breathes(newt).
has_legs(newt,4).
has_tail(newt).
amphibian(newt).

animal(octopus).
lays_eggs(octopus).
is_aquatic(octopus).
is_predator(octopus).
has_legs(octopus,8).
is_catsize(octopus).
invertebrate(octopus).

animal(opossum).
has_hair(opossum).
gives_milk(opossum).
is_predator(opossum).
is_toothed(opossum).
has_backbone(opossum).
breathes(opossum).
has_legs(opossum,4).
has_tail(opossum).
mammal(opossum).

animal(oryx).
has_hair(oryx).
gives_milk(oryx).
is_toothed(oryx).
has_backbone(oryx).
breathes(oryx).
has_legs(oryx,4).
has_tail(oryx).
is_catsize(oryx).
mammal(oryx).

animal(ostrich).
has_feathers(ostrich).
lays_eggs(ostrich).
has_backbone(ostrich).
breathes(ostrich).
has_legs(ostrich,2).
has_tail(ostrich).
is_catsize(ostrich).
bird(ostrich).

animal(parakeet).
has_feathers(parakeet).
lays_eggs(parakeet).
is_airborne(parakeet).
has_backbone(parakeet).
breathes(parakeet).
has_legs(parakeet,2).
has_tail(parakeet).
is_domestic(parakeet).
bird(parakeet).

animal(penguin).
has_feathers(penguin).
lays_eggs(penguin).
is_aquatic(penguin).
is_predator(penguin).
has_backbone(penguin).
breathes(penguin).
has_legs(penguin,2).
has_tail(penguin).
is_catsize(penguin).
bird(penguin).

animal(pheasant).
has_feathers(pheasant).
lays_eggs(pheasant).
is_airborne(pheasant).
has_backbone(pheasant).
breathes(pheasant).
has_legs(pheasant,2).
has_tail(pheasant).
bird(pheasant).

animal(pike).
lays_eggs(pike).
is_aquatic(pike).
is_predator(pike).
is_toothed(pike).
has_backbone(pike).
has_fins(pike).
has_tail(pike).
is_catsize(pike).
fish(pike).

animal(piranha).
lays_eggs(piranha).
is_aquatic(piranha).
is_predator(piranha).
is_toothed(piranha).
has_backbone(piranha).
has_fins(piranha).
has_tail(piranha).
fish(piranha).

animal(pitviper).
lays_eggs(pitviper).
is_predator(pitviper).
is_toothed(pitviper).
has_backbone(pitviper).
breathes(pitviper).
is_venomous(pitviper).
has_tail(pitviper).
reptile(pitviper).

animal(platypus).
has_hair(platypus).
lays_eggs(platypus).
gives_milk(platypus).
is_aquatic(platypus).
is_predator(platypus).
has_backbone(platypus).
breathes(platypus).
has_legs(platypus,4).
has_tail(platypus).
is_catsize(platypus).
mammal(platypus).

animal(polecat).
has_hair(polecat).
gives_milk(polecat).
is_predator(polecat).
is_toothed(polecat).
has_backbone(polecat).
breathes(polecat).
has_legs(polecat,4).
has_tail(polecat).
is_catsize(polecat).
mammal(polecat).

animal(pony).
has_hair(pony).
gives_milk(pony).
is_toothed(pony).
has_backbone(pony).
breathes(pony).
has_legs(pony,4).
has_tail(pony).
is_domestic(pony).
is_catsize(pony).
mammal(pony).

animal(porpoise).
gives_milk(porpoise).
is_aquatic(porpoise).
is_predator(porpoise).
is_toothed(porpoise).
has_backbone(porpoise).
breathes(porpoise).
has_fins(porpoise).
has_tail(porpoise).
is_catsize(porpoise).
mammal(porpoise).

animal(puma).
has_hair(puma).
gives_milk(puma).
is_predator(puma).
is_toothed(puma).
has_backbone(puma).
breathes(puma).
has_legs(puma,4).
has_tail(puma).
is_catsize(puma).
mammal(puma).

animal(pussycat).
has_hair(pussycat).
gives_milk(pussycat).
is_predator(pussycat).
is_toothed(pussycat).
has_backbone(pussycat).
breathes(pussycat).
has_legs(pussycat,4).
has_tail(pussycat).
is_domestic(pussycat).
is_catsize(pussycat).
mammal(pussycat).

animal(raccoon).
has_hair(raccoon).
gives_milk(raccoon).
is_predator(raccoon).
is_toothed(raccoon).
has_backbone(raccoon).
breathes(raccoon).
has_legs(raccoon,4).
has_tail(raccoon).
is_catsize(raccoon).
mammal(raccoon).

animal(reindeer).
has_hair(reindeer).
gives_milk(reindeer).
is_toothed(reindeer).
has_backbone(reindeer).
breathes(reindeer).
has_legs(reindeer,4).
has_tail(reindeer).
is_domestic(reindeer).
is_catsize(reindeer).
mammal(reindeer).

animal(rhea).
has_feathers(rhea).
lays_eggs(rhea).
is_predator(rhea).
has_backbone(rhea).
breathes(rhea).
has_legs(rhea,2).
has_tail(rhea).
is_catsize(rhea).
bird(rhea).

animal(scorpion).
is_predator(scorpion).
breathes(scorpion).
is_venomous(scorpion).
has_legs(scorpion,8).
has_tail(scorpion).
invertebrate(scorpion).

animal(seahorse).
lays_eggs(seahorse).
is_aquatic(seahorse).
is_toothed(seahorse).
has_backbone(seahorse).
has_fins(seahorse).
has_tail(seahorse).
fish(seahorse).

animal(seal).
has_hair(seal).
gives_milk(seal).
is_aquatic(seal).
is_predator(seal).
is_toothed(seal).
has_backbone(seal).
breathes(seal).
has_fins(seal).
is_catsize(seal).
mammal(seal).

animal(sealion).
has_hair(sealion).
gives_milk(sealion).
is_aquatic(sealion).
is_predator(sealion).
is_toothed(sealion).
has_backbone(sealion).
breathes(sealion).
has_fins(sealion).
has_legs(sealion,2).
has_tail(sealion).
is_catsize(sealion).
mammal(sealion).

animal(seasnake).
is_aquatic(seasnake).
is_predator(seasnake).
is_toothed(seasnake).
has_backbone(seasnake).
is_venomous(seasnake).
has_tail(seasnake).
reptile(seasnake).

animal(seawasp).
lays_eggs(seawasp).
is_aquatic(seawasp).
is_predator(seawasp).
is_venomous(seawasp).
invertebrate(seawasp).

animal(skimmer).
has_feathers(skimmer).
lays_eggs(skimmer).
is_airborne(skimmer).
is_aquatic(skimmer).
is_predator(skimmer).
has_backbone(skimmer).
breathes(skimmer).
has_legs(skimmer,2).
has_tail(skimmer).
bird(skimmer).

animal(skua).
has_feathers(skua).
lays_eggs(skua).
is_airborne(skua).
is_aquatic(skua).
is_predator(skua).
has_backbone(skua).
breathes(skua).
has_legs(skua,2).
has_tail(skua).
bird(skua).

animal(slowworm).
lays_eggs(slowworm).
is_predator(slowworm).
is_toothed(slowworm).
has_backbone(slowworm).
breathes(slowworm).
has_tail(slowworm).
reptile(slowworm).

animal(slug).
lays_eggs(slug).
breathes(slug).
invertebrate(slug).

animal(sole).
lays_eggs(sole).
is_aquatic(sole).
is_toothed(sole).
has_backbone(sole).
has_fins(sole).
has_tail(sole).
fish(sole).

animal(sparrow).
has_feathers(sparrow).
lays_eggs(sparrow).
is_airborne(sparrow).
has_backbone(sparrow).
breathes(sparrow).
has_legs(sparrow,2).
has_tail(sparrow).
bird(sparrow).

animal(squirrel).
has_hair(squirrel).
gives_milk(squirrel).
is_toothed(squirrel).
has_backbone(squirrel).
breathes(squirrel).
has_legs(squirrel,2).
has_tail(squirrel).
mammal(squirrel).

animal(starfish).
lays_eggs(starfish).
is_aquatic(starfish).
is_predator(starfish).
has_legs(starfish,5).
invertebrate(starfish).

animal(stingray).
lays_eggs(stingray).
is_aquatic(stingray).
is_predator(stingray).
is_toothed(stingray).
has_backbone(stingray).
is_venomous(stingray).
has_fins(stingray).
has_tail(stingray).
is_catsize(stingray).
fish(stingray).

animal(swan).
has_feathers(swan).
lays_eggs(swan).
is_airborne(swan).
is_aquatic(swan).
has_backbone(swan).
breathes(swan).
has_legs(swan,2).
has_tail(swan).
is_catsize(swan).
bird(swan).

animal(termite).
lays_eggs(termite).
breathes(termite).
has_legs(termite,6).
insect(termite).

animal(toad).
lays_eggs(toad).
is_aquatic(toad).
is_toothed(toad).
has_backbone(toad).
breathes(toad).
has_legs(toad,4).
amphibian(toad).

animal(tortoise).
lays_eggs(tortoise).
has_backbone(tortoise).
breathes(tortoise).
has_legs(tortoise,4).
has_tail(tortoise).
is_catsize(tortoise).
reptile(tortoise).

animal(tuatara).
lays_eggs(tuatara).
is_predator(tuatara).
is_toothed(tuatara).
has_backbone(tuatara).
breathes(tuatara).
has_legs(tuatara,4).
has_tail(tuatara).
reptile(tuatara).

animal(tuna).
lays_eggs(tuna).
is_aquatic(tuna).
is_predator(tuna).
is_toothed(tuna).
has_backbone(tuna).
has_fins(tuna).
has_tail(tuna).
is_catsize(tuna).
fish(tuna).

animal(vampire).
has_hair(vampire).
gives_milk(vampire).
is_airborne(vampire).
is_toothed(vampire).
has_backbone(vampire).
breathes(vampire).
has_legs(vampire,2).
has_tail(vampire).
mammal(vampire).

animal(vole).
has_hair(vole).
gives_milk(vole).
is_toothed(vole).
has_backbone(vole).
breathes(vole).
has_legs(vole,4).
has_tail(vole).
mammal(vole).

animal(vulture).
has_feathers(vulture).
lays_eggs(vulture).
is_airborne(vulture).
is_predator(vulture).
has_backbone(vulture).
breathes(vulture).
has_legs(vulture,2).
has_tail(vulture).
is_catsize(vulture).
bird(vulture).

animal(wallaby).
has_hair(wallaby).
gives_milk(wallaby).
is_toothed(wallaby).
has_backbone(wallaby).
breathes(wallaby).
has_legs(wallaby,2).
has_tail(wallaby).
is_catsize(wallaby).
mammal(wallaby).

animal(wasp).
has_hair(wasp).
lays_eggs(wasp).
is_airborne(wasp).
breathes(wasp).
is_venomous(wasp).
has_legs(wasp,6).
insect(wasp).

animal(wolf).
has_hair(wolf).
gives_milk(wolf).
is_predator(wolf).
is_toothed(wolf).
has_backbone(wolf).
breathes(wolf).
has_legs(wolf,4).
has_tail(wolf).
is_catsize(wolf).
mammal(wolf).

animal(worm).
lays_eggs(worm).
breathes(worm).
invertebrate(worm).

animal(wren).
has_feathers(wren).
lays_eggs(wren).
is_airborne(wren).
has_backbone(wren).
breathes(wren).
has_legs(wren,2).
has_tail(wren).
bird(wren).

