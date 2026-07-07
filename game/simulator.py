"""Small two-player Briscola engine"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from game.cards import Card, Suit, full_deck

PlayerId = int


@dataclass(frozen=True, slots=True)
class PlayedCard:
    player: PlayerId
    card: Card


@dataclass(frozen=True, slots=True)
class DrawEvent:
    player: PlayerId
    card: Card
    was_visible_trump: bool


@dataclass(frozen=True, slots=True)
class TrickResult:
    leader: PlayerId
    winner: PlayerId
    cards: tuple[PlayedCard, PlayedCard]
    points: int
    scores_after: tuple[int, int]
    draws: tuple[DrawEvent, ...]


@dataclass(frozen=True, slots=True)
class PublicState:
    trump_suit: Suit
    trump_card: Card
    trump_card_in_stock: bool
    leader: PlayerId
    current_player: PlayerId | None
    current_trick: tuple[PlayedCard, ...]
    played_cards: tuple[PlayedCard, ...]
    scores: tuple[int, int]
    stock_size: int
    hand_sizes: tuple[int, int]
    tricks_played: int
    finished: bool


@dataclass(frozen=True, slots=True)
class PlayerView:
    player: PlayerId
    hand: tuple[Card, ...]
    public_state: PublicState


def _validate_player(player: PlayerId) -> None:
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player!r}")


def _other_player(player: PlayerId) -> PlayerId:
    return 1 - player


def trick_winner(
    first: PlayedCard,
    second: PlayedCard,
    trump_suit: Suit,
) -> PlayerId:
    """Decide who wins a two-card trick"""

    lead_card = first.card
    follow_card = second.card

    if follow_card.suit == lead_card.suit:
        if follow_card.strength > lead_card.strength:
            return second.player
        return first.player

    if follow_card.suit == trump_suit and lead_card.suit != trump_suit:
        return second.player

    return first.player


class BriscolaGame:
    """State for one two-player game"""

    def __init__(
        self,
        hands: tuple[list[Card], list[Card]],
        stock: list[Card],
        trump_card: Card,
        first_player: PlayerId = 0,
    ) -> None:
        _validate_player(first_player)
        self.hands = hands
        self.stock = stock
        self.trump_card = trump_card
        self.trump_suit = trump_card.suit
        self.leader = first_player
        self.current_player = first_player
        self.current_trick: list[PlayedCard] = []
        self.played_cards: list[PlayedCard] = []
        self.captured_cards: tuple[list[Card], list[Card]] = ([], [])
        self.scores = [0, 0]
        self.trick_history: list[TrickResult] = []

    @classmethod
    def new(cls, seed: int | None = None, first_player: PlayerId = 0) -> "BriscolaGame":
        """Start a shuffled game"""

        rng = random.Random(seed)
        cards = list(full_deck())
        rng.shuffle(cards)
        return cls.from_deck(cards, first_player=first_player)

    @classmethod
    def from_deck(
        cls,
        deck: Sequence[Card],
        first_player: PlayerId = 0,
    ) -> "BriscolaGame":
        """Start a game from a known deck order"""

        cards = list(deck)
        expected_deck = set(full_deck())
        if len(cards) != 40 or set(cards) != expected_deck:
            raise ValueError("deck must contain exactly the 40 unique Briscola cards")

        hands = (cards[0:3], cards[3:6])
        trump_card = cards[6]
        # The visible trump is drawn last.
        stock = cards[7:] + [trump_card]
        return cls(hands=hands, stock=stock, trump_card=trump_card, first_player=first_player)

    def legal_moves(self, player: PlayerId | None = None) -> tuple[Card, ...]:
        """Cards the player can play right now"""

        if player is None:
            player = self.current_player
        if player is None:
            return ()
        _validate_player(player)
        if self.finished:
            return ()
        if player != self.current_player:
            return ()
        # Briscola has no follow-suit constraint.
        return tuple(self.hands[player])

    def play_card(self, player: PlayerId, card: Card) -> TrickResult | None:
        """Play a card, returning a result only after the second card"""

        _validate_player(player)
        if self.finished:
            raise RuntimeError("cannot play a card after the game is finished")
        if player != self.current_player:
            raise ValueError(f"it is player {self.current_player}'s turn, not player {player}")
        if card not in self.hands[player]:
            raise ValueError(f"player {player} does not hold {card}")
        if len(self.current_trick) >= 2:
            raise RuntimeError("current trick is already complete")

        self.hands[player].remove(card)
        self.current_trick.append(PlayedCard(player=player, card=card))

        if len(self.current_trick) == 1:
            self.current_player = _other_player(player)
            return None

        return self._complete_trick()

    def public_state(self) -> PublicState:
        """Information that both players can see"""

        return PublicState(
            trump_suit=self.trump_suit,
            trump_card=self.trump_card,
            trump_card_in_stock=self.trump_card in self.stock,
            leader=self.leader,
            current_player=None if self.finished else self.current_player,
            current_trick=tuple(self.current_trick),
            played_cards=tuple(self.played_cards),
            scores=(self.scores[0], self.scores[1]),
            stock_size=len(self.stock),
            hand_sizes=(len(self.hands[0]), len(self.hands[1])),
            tricks_played=len(self.trick_history),
            finished=self.finished,
        )

    def player_view(self, player: PlayerId) -> PlayerView:
        """Public state plus this player's own hand"""

        _validate_player(player)
        return PlayerView(
            player=player,
            hand=tuple(self.hands[player]),
            public_state=self.public_state(),
        )

    @property
    def finished(self) -> bool:
        return (
            not self.stock
            and not self.hands[0]
            and not self.hands[1]
            and not self.current_trick
        )

    def winner(self) -> PlayerId | None:
        """Winner after the game ends; None means a tie"""

        if not self.finished:
            raise RuntimeError("winner is available only after the game is finished")
        if self.scores[0] == self.scores[1]:
            return None
        return 0 if self.scores[0] > self.scores[1] else 1

    def _complete_trick(self) -> TrickResult:
        first, second = self.current_trick
        winner = trick_winner(first, second, self.trump_suit)
        points = first.card.points + second.card.points

        self.played_cards.extend((first, second))
        self.captured_cards[winner].extend((first.card, second.card))
        self.scores[winner] += points

        draws = self._draw_after_trick(winner)
        result = TrickResult(
            leader=self.leader,
            winner=winner,
            cards=(first, second),
            points=points,
            scores_after=(self.scores[0], self.scores[1]),
            draws=draws,
        )

        self.current_trick.clear()
        self.leader = winner
        self.current_player = winner
        self.trick_history.append(result)
        return result

    def _draw_after_trick(self, winner: PlayerId) -> tuple[DrawEvent, ...]:
        draws: list[DrawEvent] = []
        for player in (winner, _other_player(winner)):
            if not self.stock:
                break
            card = self.stock.pop(0)
            self.hands[player].append(card)
            draws.append(
                DrawEvent(
                    player=player,
                    card=card,
                    was_visible_trump=card == self.trump_card,
                )
            )
        return tuple(draws)
