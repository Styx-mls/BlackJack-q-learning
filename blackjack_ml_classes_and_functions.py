import random
import pickle

class Player:
    """
    Represents a player in the card game. Handles the player's hand and related actions.
    """

    def __init__(self):
        """
        Initializes a player with an empty hand and a 'not busted' state.
        """
        self.busted = False
        self.hand = []

    def hit(self, deck):
        """
        Adds a card from the deck to the player's hand.
        Args:
            deck (Deck): The deck to deal a card from.
        """
        self.hand.append(deck.deal_card())

    def show_hand(self):
        """
        Prints the player's hand in a readable format.
        """
        for i in range(len(self.hand)):
            print(self.hand[i], end=" ")

    def get_hand_sum(self):
        """
        Calculates the total value of the player's hand, accounting for Aces.

        Returns:
            int: The total value of the player's hand.
        """
        hand_sum = 0
        aces = 0
        
        for i in range(len(self.hand)):
            if self.hand[i].rank.isalpha() and self.hand[i].rank != "A":
                hand_sum += 10  # Face cards count as 10
            elif self.hand[i].rank == "A":
                hand_sum += 11  # Aces initially count as 11
                aces += 1
            else:
                hand_sum += int(self.hand[i].rank)  # Number cards are added directly

        # Adjust Aces if the total exceeds 21
        while hand_sum > 21 and aces:
            hand_sum -= 10
            aces -= 1

        return hand_sum

    def is_busted(self):
        """
        Updates the player's 'busted' status based on their hand value.
        """
        if self.get_hand_sum() > 21:
            self.busted = True

class AI_Player(Player):
    """
    Represents an AI player that learns and makes decisions using Q-learning.
    """

    def __init__(self, learning_rate=0.3, discount_factor=0.95, exploration_rate=1.0):
        """
        Initializes the AI player with Q-learning parameters and an empty Q-table.
        """
        super().__init__()
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def get_state(self, player_sum, dealer_card):
        """
        Constructs a state representation for Q-learning.

        Args:
            player_sum (int): The player's hand sum.
            dealer_card (Card): The dealer's visible card.

        Returns:
            tuple: A state representation as (player_sum, rank_dealer_card).
        """
        rank_dealer_card = None

        if dealer_card.rank.isalpha() and dealer_card.rank != 'A':
            rank_dealer_card = 10
        elif dealer_card.rank == 'A':
            rank_dealer_card = 11
        else:
            rank_dealer_card = int(dealer_card.rank)

        return (player_sum, rank_dealer_card)

    def choose_action(self, state):
        """
        Chooses an action (hit or stand) based on exploration and Q-values.

        Args:
            state (tuple): The current state.

        Returns:
            str: 'hit' or 'stand'.
        """
        if random.random() < self.epsilon:
            return random.choice(['hit', 'stand'])  # Random action (exploration)

        if self.q_table.get(state, [0, 0])[0] >= self.q_table.get(state, [0, 0])[1]:
            return 'hit'
        else:
            return 'stand'

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-value for a given state-action pair.

        Args:
            state (tuple): The current state.
            action (str): The action taken ('hit' or 'stand').
            reward (float): The reward received.
            next_state (tuple or None): The next state.
        """
        action_index = 0 if action == 'hit' else 1

        next_max = max(self.q_table.get(next_state, [0, 0])) if next_state else 0
        old_q = self.q_table.get(state, [0, 0])[action_index]
        new_value = old_q + self.alpha * (reward + self.gamma * next_max - old_q)

        if state not in self.q_table:
            self.q_table[state] = [0, 0]

        self.q_table[state][action_index] = new_value

    def decay_exploration(self, decay_rate):
        """
        Reduces the exploration rate to encourage exploitation over time.

        Args:
            decay_rate (float): The decay rate for epsilon.
        """
        self.epsilon = max(0.1, self.epsilon * decay_rate)

    def train_module(self, n_games=1000):
        """
        Trains the AI player over a specified number of simulated games.

        Args:
            n_games (int): The number of training games.
        """
        for _ in range(n_games):
            self.hand = []
            self.busted = False
            deck = Deck()
            dealer = Dealer()
            deck.shuffle_deck()
            self.hit(deck)
            self.hit(deck)
            dealer.hit(deck)
            dealer.hit(deck)

            state = self.get_state(self.get_hand_sum(), dealer.hand[0])

            while not self.busted:
                choice = self.choose_action(state)

                if choice == 'stand':
                    break

                self.hit(deck)
                self.is_busted()
                reward = None

                if self.busted:
                    reward = -1
                    next_state = None
                else:
                    reward = 1
                    next_state = self.get_state(self.get_hand_sum(), dealer.hand[0])

                self.update_q_value(state, choice, reward, next_state)
                state = next_state

            if self.busted:
                pass

            elif dealer.get_hand_sum() > self.get_hand_sum():
                reward = -0.5
                self.update_q_value(state, choice, reward, None)

            elif dealer.get_hand_sum() == self.get_hand_sum():
                reward = 0.5
                self.update_q_value(state, choice, reward, None)

            else:
                reward = 1
                self.update_q_value(state, choice, reward, None)

            self.decay_exploration(0.9999)

class Dealer(Player):
    """
    Represents the dealer in the card game. Extends the Player class.
    """

    def should_hit(self):
        """
        Determines if the dealer should hit based on their hand value.

        Returns:
            bool: True if the dealer should hit, False otherwise.
        """
        return self.get_hand_sum() <= 17

    def show_first_card(self):
        """
        Displays the dealer's first card.
        """
        print(self.hand[0])

class Card:
    """
    Represents a single playing card with a rank and suit.
    """

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return self.rank + self.suit

class Deck:
    """
    Represents a standard deck of 52 playing cards.
    """

    def __init__(self):
        suits = ['\u2660', '\u2661', '\u2662', '\u2663']  # Spades, Hearts, Diamonds, Clubs
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(rank, suit) for suit in suits for rank in ranks]

    def shuffle_deck(self):
        """
        Shuffles the deck randomly.
        """
        random.shuffle(self.cards)

    def deal_card(self):
        """
        Deals the top card from the deck.

        Returns:
            Card: The top card from the deck.
        """
        return self.cards.pop(0)

    def __str__(self):
        return ' '.join(str(card) for card in self.cards)

def get_user_input():
    """
    Prompts the user to enter 'yes' or 'no'.

    Returns:
        str: User's input.
    """
    query = input("Do you wish to watch again? ").strip().lower()
    while query not in ['yes', 'no']:
        query = input("Invalid input. Please enter 'yes' or 'no': ").strip().lower()
    return query

def play_game():
    """
    Simulates a game between the AI player and the dealer.
    """
    query = 'yes'

    with open("training_bot.pkl", "rb") as bot_file:
        p1 = pickle.load(bot_file)

    while query == 'yes':
        deck = Deck()
        dealer = Dealer()
        deck.shuffle_deck()
        p1.hand = []
        p1.busted = False
        p1.epsilon = 0
        p1.hit(deck)
        p1.hit(deck)
        dealer.hit(deck)
        dealer.hit(deck)

        print("AI player's cards are: ", end=" ")
        p1.show_hand()
        print()
        print("The dealer's first card is: ", end=" ")
        dealer.show_first_card()

        state = p1.get_state(p1.get_hand_sum(), dealer.hand[0])

        while not p1.busted and p1.choose_action(state) == 'hit':
            p1.hit(deck)
            p1.is_busted()
            print("AI player's cards are now: ", end=" ")
            p1.show_hand()
            print()
            if p1.busted:
                print("Bot busted out.")
            state = p1.get_state(p1.get_hand_sum(), dealer.hand[0])

        if not p1.busted:
            while not dealer.busted and dealer.should_hit():
                dealer.hit(deck)
                dealer.is_busted()

            if dealer.busted:
                print("Dealer's hand: ", end="")
                dealer.show_hand()
                print("\nDealer busted out. Bot wins!")
            elif dealer.get_hand_sum() > p1.get_hand_sum():
                print("Dealer's hand: ", end="")
                dealer.show_hand()
                print("\nDealer wins!")
            elif dealer.get_hand_sum() == p1.get_hand_sum():
                print("Dealer's hand: ", end="")
                dealer.show_hand()
                print("\nTie!")
            else:
                print("Dealer's hand: ", end="")
                dealer.show_hand()
                print("\nBot wins!")

        query = get_user_input()
