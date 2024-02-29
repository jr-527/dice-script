help_string = '''
Available things:
 +, -, *, / (, ):
   Does what you'd expect, eg 2d6-1d4+3, 3*1d6 (not the same as 3d6), 1d4*(2d6+2), 8d6/2.
   Division (or multiplying by a fraction) rounds towards 0.
   You can also do things like (1d4)d6, which represents rolling 1d4 then adding up that many d6.

 ^, **:
   Raise to a power, eg 1d3^2 is either 1, 4, or 9, 1d4**2 is either 1, 4, 9, or 16.

 adv, dis:
   Gives the better/worse of 2 attempts, eg adv(1d20), dis(3d6).

 highest, lowest:
   Like adv() or dis() but with other numbers, eg highest(1d20, 3) is the best of 3 rolls.
   Syntax: highest(die, number of rolls), lowest(die, number of rolls)

 min0, min1, min_val:
   min0, min1 mean results less than 0 or 1 are replaced with 0 or 1 respectively.
   min_val does the same but with an arbitrary minimum, so in min_val(1d6-3, 2),
   results less than 2 are replaced with 2.
   Syntax: min0(die), min1(die), min_val(die, minimum)

 >, >=, <, <=, ==, !=:
   3d4 > 2d6 prints the probability that 3d4 > 2d6, and the other symbols are similar.
   Returns a probability distribution, where 1 represents true, 0 represents false.
   You can chain these, ie 2 <= 1d20 <= 19 gives the probability of 1d20 giving between
   2 and 19, 1d4 <= 1d20 < 12 the probability of rolling 1d4 and 1d20 and having the 1d20
   be between the 1d4 (inclusive) and 12 (exclusive)

 4d6dl or '4d6 drop lowest':
   This gives the distribution of 4d6 with the lowest die removed.

 @:
   More advanced version of (1d4)d6. If an attack hits 1d4+2 times, and each hit
   deals 2d10+1d4 damage, then you can write that as (1d4+2) @ (2d10+1d4).
   Syntax: thing1 @ thing2. Gives the distribution of evaluating thing1 then adding
           up that many copies of thing2.

There are also more advanced functions that implement things like checks/saves, attacks, rerolling,
and some logic. Type "help advanced" to see instructions for these advanced functions.'''

help_advanced = '''
Advanced functions:
 order:
   Like highest or lowest, but it can give middle values.
   Syntax: order(die, number of rolls, which roll to keep), where 1 means keep the lowest roll.
   Ex: order(4d6dl, 6, 2) gives your 2nd lowest roll from doing 4d6 drop lowest 6 times.

 check, save:
   This gives the probability of a DnD 5e ability check succeeding.
   Syntax 1: check(bonus, dc, [adv])
   adv is optional.
   Set adv to True or 1 for advantage, 2 for double advantage, 3 for triple, etc
   -1 for disadvantage, -2 for double disadvantage, etc.
   Ex: check(6+1d4, 16)
   Ex: check(6, 16, True)
   Syntax 2: check(bonus, dc, [adv], succeed=die1, fail=die2)
   This gives the distribution of making the check then rolling die1 if it passes
   and die2 if it fails.
   Ex: save(6, 16, succeed=8d6/2, fail=8d6)

 attack or crit:
   This calculates everything related to a DnD 5e attack roll, including crits, expanded
   crit range, extra crit damage
   Basic syntax: crit(attack_bonus, enemy_ac, damage_dice, damage_bonus)
   Ex: attack(4, 16, 2d6, 3) for an attack with +4 to hit that deals 2d6+3 vs AC 16
   Ex: crit(4+1d4, 16, 2d6, 3, adv=True) for an attack with advantage, with
       +4+1d4 to hit, dealing 2d6+3 vs AC 16
   Enter "help attack" for more advanced usage of this function.

 reroll:
   Rerolls if the dice roll comes out to certain values, keeping the new result.
   You can manually specify numbers to reroll, give upper or lower bounds for numbers
   to reroll, or do a combination of both.
   Syntax: reroll(die, [value1_to_reroll, value2_to_reroll, ..., lower=?, upper=?])
   The things in [...] are optional.
   Ex: reroll(1d6, 1, 2) or reroll(1d6, upper=2) rolls 1d6, rerolling 1s or 2s.
   Ex: reroll(1d20, 13, lower=5, upper=9) rolls 1d20, rerolling 5, 6, 7, 8, 9, or 13.

 choice:
   Function for if/else statements. Multiple possible syntaxes.
   Syntax 1: choice(condition, if_false, if_true)
   Syntax 2: choice(probability_of_true, if_false, if_true)
   Syntax 3: choice(distribution, value1, value2, ...)
   Enter "help choice" for other ways to use choice.

 help:
   If called as a function, help prints another function's Python documentation,
   eg help(min0). Only recommended for people familiar with Python.
   To be precise, terminal input other than "help x" goes through some regex
   processing and is then passed to Python's eval function, so you can generally
   input arbitrary Python expressions.'''

choice_help = '''
Syntax 1: choice(condition, if_false, if_true)
Gives the distribution of (if_true if condition is true, otherwise if_false)
Ex: choice(1d20+3>=15, 8d6, 8d6/2) is the distribution of the following process:
1. roll 1d20+3
2. if the result is at least 15, return 8d6/2
3. otherwise return 8d6
This models a +3 dex save against fireball with DC 15

Syntax 2: choice(probability_of_true, if_false, if_true)
This gives (if_true with probability probability_of_true, otherwise if_false).
Ex: choice(.6, 1d5, 0) is the distribution of
(0 with probability .6 and 1d5 with probability .4)

Syntax 3: choice(dist, value1, value2, ...)
This gives the distribution of
(value1 if distr gives its first value, value2 if distr gives its second value, etc)
distr can be a die expression or an array.
Ex: choice(2d2, 1d6, 1d8, 1d10) gives the distribution of rolling 2d2 and
returning 1d6 if you got 2, 1d8 if you got 3, and 1d10 if you got 4.
Ex: choice([.2, .5, .3], 1d4, 1d6, 1d8) means 
1d4 with probability .2, 1d6 with probability .5, and 1d8 with probability .3'''

attack_help = '''
Syntax:
attack(bonus, ac, damage, damage_bonus, [extra_dice, crit_range, adv, no_crit_damage])
The arguments in [...] are optional but must be supplied by name, after all
of the mandatory arguments, ie:
attack(4, 16, 1d6, 3, crit_range=19, adv=True).

Arguments:
bonus: The attack bonus, ie 4 or 4+1d4
ac: The target AC, ie 16 or 14+1d8
damage: The damage dice, ie 2d6
damage_bonus: The fixed number always added to the damage, ie 3
extra_dice: Additional dice added to crits, for features like brutal critical
crit_range: The attack crits on a d20 roll of this number or higher
adv: 1 or True for advantage, 2 for double advantage, 3 for triple advantage, etc.
     -1 for disadvantage, -2 for double disadvantage, etc
no_crit_damage: If True, all bonus damage for crits is blocked. This means that
                crits auto-hit but deal regular attack damage.'''