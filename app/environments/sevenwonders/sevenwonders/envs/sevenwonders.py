from .Players import Player
from .common import *
from random import randint, choice, shuffle
from .Wonders import *
from .Cards import *
import gym
from gym import spaces
import numpy as np
import json
import os
from stable_baselines import logger

class SevenWondersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, log_results= False, verbose=False, manual=False):
        super(SevenWondersEnv, self).__init__()
        
        self.name = "sevenwonders"
        self.manual = manual
        self.n_players = 3
        self.card_types = 75
        self.action_space = gym.spaces.Discrete(151)
        self.observation_space = gym.spaces.Box(0,1,(386,))
        self.verbose = verbose
        self.log_results = log_results
        
        self.alt_reward =False

        

    @property
    def observation(self):
        obs = np.zeros(386)
        player = self.current_player

        
        obs[:151] = self.legal_actions
            
        for card in player.tableau: #obs 89-163 (cards built)
            obs[151+card.id] = 1

        obs[226] = self.age/3  #Current age
        obs[227] = min(player.resources[RESOURCE_GOLD]/20.0,1) #gold over 20 is just useless
        obs[228] = player.wonder.id/6.0 #Wonder id
        obs[229] = player.wonder.stages_completed/4.0 #Wonder stages completed
        obs[230] = player.shields/25.0    #War shields
        obs[231] = player.east_player.shields/25.0    #East war shields
        obs[232] = player.west_player.shields/25.0    #West war shields
        obs[233] = int(player.has_double_last_cards)
        
        for card in player.east_player.tableau:
            obs[234+card.id] = 1 #obs 172-246 (cards built by east player)
        for card in player.west_player.tableau:
            obs[309+card.id] = 1 #obs 247-321 (cards built by west player)
        obs[384] = 1 if self.turn_type == 1 else 0 #Is it a wonder turn?
        obs[385] = 1 if self.turn_type == 2 else 0 #Is it a Halikarnassos turn?
        
        return obs

    @property
    def legal_actions(self):
        legal_actions = np.zeros(151)
        player = self.current_player
        
        if self.turn_type != 2: #normal turn or wonder discard
            if self.turn_type == 0:
                wonder_available = False
                if not player.wonder.all_done and player.get_price(player.wonder) != -1:
                    wonder_available = True
                legal_actions[0] += int(wonder_available)

            for card in player.hand:
                card_id = card.id
                if self.turn_type == 0 and player.get_price(card)!=-1:  
                    legal_actions[card_id+1] = 1 

                legal_actions[76+card_id] = 1 
        else:
            for card in self.discard: 
                legal_actions[card.id+1] = 1 
            
        #print(legal_actions)
        return legal_actions

    @property
    def current_player(self):
        return self.players[self.current_player_num] 
    

    def switch_hands(self):
        logger.debug(f'\nSwitching hands...')
        playernhand = self.players[-1].hand

        for i in range(self.n_players - 1, -1, -1):
            if i > 0:
                self.players[i].hand = self.players[i-1].hand


        self.players[0].hand = playernhand

    def step(self, action): 
        """Each step is a player turn (or a part of it if they have 2 actions)
        Special turns: 
            1) pick a card for wonder 
            2) pick a card after Halikarnassos
            3) pick second last card
                - if not none, after last player turn, go back to this player
                - observation is the same as normal turn
        """
        
        #print(f"\n{self.current_player.name}'s turn\n")
        #print(f'sevenwonders.py action {action} chosen')
        reward = [0] * self.n_players
        done = False
        player = self.current_player

         
        # check move legality
        if self.legal_actions[action] == 0: #Illegal move
            logger.debug(f'\n{player} tried an illegal move, ending game')
            #print(action)
            #print(f'\n{player.name} tried an illegal move, ending game')
            #print("legal: ",self.legal_actions)
            reward = [1.0/(self.n_players-1)] * self.n_players
            reward[self.current_player_num] = -1
            done = True

        
    
        elif self.turn_type == 1: #wonder turn, choose which card to tuck
            action = self.translate_action(action,player)   
            logger.debug(player.choose_card_for_wonder(action))
            if self.turn == 6 and player.hand:
                self.discard.append(player.hand.pop())

        elif self.turn_type == 2 and self.discard: #Halikarnassos turn, choose discarded card to play
            card = player.play_from_discard(action,self.discard)
            logger.debug(f'\n{player} played {card} from discard pile') 
            

        else: #Legal move, normal turn
            action = self.translate_action(action,player)   
            if action == 0:
                logger.debug(f'\n{player} played wonder {player.wonder.name} stage {player.wonder.stages_completed+1}')
                price = player.get_price(player.wonder)
                if price !=0:
                    player.resources[RESOURCE_GOLD] -= price['east']
                    player.resources[RESOURCE_GOLD] -= price['west']
                self.next_player.append((self.current_player_num,1)) #keep the current player
                self.action_bank.append((player.wonder.name,player,price))
                #If babylon b stage 2 is played on 6th turn, make 7th card available to play
                if player.wonder.name == "Babylon" and \
                    self.turn == 6 and \
                    player.wonder.side == "B" and \
                    player.wonder.stages_completed == 1:
                    self.next_player.append((self.current_player_num,3))
                    
            
            else:
                if action % 2 == 1: #Add card to play queue
                    card = player.hand.pop(action//2)
                   
                    """Used only now"""
                    #Reward function
                    #if card.color == COLOR_GREEN:
                       #reward[self.current_player_num] += 0.05
                

                    price = player.get_price(card)
                    logger.debug(f'\n{player} played {card}')
                    logger.debug(f'\ncost = {price}')  
                    
                    self.action_bank.append((card,player,price))
                    #print("played: "+str(card))
                    if price !=0:
                        if price == 1:
                            player.resources[RESOURCE_GOLD] -= 1
                        else: #Pay other players
                            player.resources[RESOURCE_GOLD] -= price['east']
                            player.resources[RESOURCE_GOLD] -= price['west']

                else: #discard card (that could be played)
                    card = player.hand.pop(action//2 - 1)
                    
                    """Used only now"""
                    #Reward function
                    #reward[self.current_player_num] -= 0.1
                    

                    logger.debug(f'\n{player} discarded {card} for 3 coins')
                    player.resources[RESOURCE_GOLD] += 3
                    self.discard.insert(0,card)
                    #print("discarded: "+str(card))
                
                if self.turn == 6:
                    if player.has_double_last_cards and self.turn_type != 3: #If played first card from 2 last cards
                        logger.debug(f'\n{player} can play double last card')
                        self.next_player.append((self.current_player_num,3))
                    elif (self.current_player_num,1) not in self.next_player and not player.has_double_last_cards:
                            self.discard.append(player.hand.pop()) #discard last card
                            

            
            
            
        if not self.next_player: #if all players have played (except halikarnassos)
            logger.debug("\neveryone has played, resolving card/wonder effects")
            for card,player,cost in self.action_bank: 
                if card in {"Ephesos","Rhodos","Alexandria","Babylon","Gizah"}:
                    player.play_wonder(cost)
                elif card == "Halikarnassos":
                    player.play_wonder(cost)
                    if self.discard:
                        self.next_player.insert(0,(player.id,2)) 
                else:
                    player.play_card(card,cost)
            self.action_bank.clear()
                
        
        if not self.next_player: #If end of turn (Meaning no halikarnassos)
            self.turn += 1
            for i in range(self.n_players)[::-1]: #ex: [(3,0),(2,0),(1,0),(0,0)]
                self.next_player.append((i,0))
            self.switch_hands()

        if self.turn > 6: #if end of age
            logger.debug("\nEnd of age, war time")
            #print("\nEnd of age, war time")
            #print(self.players)
            #print(self.age)
            for player in self.players: 
                player.war((self.age*2)-1) #1, 3 and 5

            if self.age == 3: #end of game
                
                logger.debug(f'\n\n---- GAME OVER ----')
                reward = self.score_game() 
                done = True
                
            
            else: #go to next age
                self.turn = 1
                self.age += 1
                self.assign_cards(self.age)
                logger.debug(f'\n\n---- AGE {self.age} ----')
        
        self.current_player_num,self.turn_type = self.next_player.pop() #Get who the next player is
        self.done = done
        #print("successful turn")
        return self.observation, reward, done, {}

    def results_to_file(self,reward):
        result = dict()
        path = "./results"
        with open(os.path.join(path, self.generate_filename(path,"game_result",".json")), "w") as json_file:
            for i, player in enumerate(self.players):
                win = 0
                if reward[i] > 0:
                    win = 1

                result[player.name] = player.get_player_log(win) 
            json.dump(result, json_file)


    def generate_filename(self,folder_path, base_name, extension):
        existing_filenames = os.listdir(folder_path)
        filename = base_name+"_"+str(len(existing_filenames)) + extension
        return filename


    def translate_action(self,action,player):
        if action == 0: #playing a wonder
            return 0
        elif action < 76: #playing a card
            for i,card in enumerate(player.hand):
                if card.id == action-1:
                    return i*2 + 1
        else: #discarding
            for i,card in enumerate(player.hand):
                if card.id == action-76:
                    return (i+1)*2


    def reset(self): 
        self.age = 1
        self.turn = 1
        self.discard = []
        self.action_bank = []


        #This next player list is a queue that determine who plays next
        #Entries are always (player#, turnType)
        #0 <= player# < n_players 
        #turn types: 
        # 0 = normal turn (this counts the play both last cards)
        # 1 = pick card for wonder turn
        # 2 = halikarnassos turn
        #ex: [(3,0),(2,0),(1,0),(0,0)] -> player 0 plays a normal turn next
        self.next_player = [(i,0) for i in range(self.n_players)][::-1]

        self.players = self.players_setup()
        self.assign_cards(self.age)

        self.current_player_num,self.turn_type = self.next_player.pop() #first player 0 normal turn
        #print(self.current_player_num)
        self.done = False
        #print(f'\n\n---- NEW GAME ----')
        logger.debug(f'\n\n---- NEW GAME ----')
        logger.debug(f'\n\n---- AGE {self.age} ----')
        return self.observation

    
    def render(self, mode='human', close=False):
        
        if close:
            return

        logger.debug(f'\n\n-------AGE {self.age} : TURN {self.turn}-----------')
        logger.debug(f"It is Player {self.current_player.id}'s turn to choose")
            

        for p in self.players:
            logger.debug(f'\n{p.name}\'s hand')
            if len(p.hand) > 0:
                logger.debug(str(p.hand))
            else:
                logger.debug('Empty')

            logger.debug(f'{p.name}\'s tableau')
            if len(p.tableau) > 0:
                logger.debug(str(p.tableau))
            else:
                logger.debug('Empty')

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
            

        
    
    
    
    
    """Game logic, nothing to do with gym"""
    
    def players_setup(self): #max 6
        wonders_list = [Alexandria,Babylon,Ephesos,Gizah,Halikarnassos,Rhodos]
        playerlist = []
        names = ["Jakob","Dara","Nicholas"]
        for i in range(self.n_players):
            #player = Player(f"Player {i+1}",i) 
            player = Player(f"{names[i]}",i)
            self.set_wonder(player,wonders_list)
            playerlist.append(player)

        for i in range(self.n_players):
            playerlist[i].set_east_player(playerlist[i-1])
            playerlist[i].set_west_player(playerlist[(i+1)%self.n_players])

        return playerlist
    
    def set_wonder(self,player,wonders_list):
        wonder = wonders_list.pop(randint(0,len(wonders_list)-1))
        player.set_wonder(wonder(player) if wonder != Halikarnassos else wonder(player,self))
        #print(player.name,player.wonder.name,player.wonder.side)
    

    def switch_hands(self):
        player = self.players[0]
        temp_hand = player.hand
        if self.age == 2: #for age 2, counterclockwise switching
            for _ in range(len(self.players)-1):
                player.hand = player.west_player.hand
                player = player.west_player
            player.hand = temp_hand    
        else: #for age 1 and 3, clockwise switching
            for _ in range(len(self.players)-1):
                player.hand = player.east_player.hand
                player = player.east_player
            player.hand = temp_hand


    def assign_cards(self, age=1):
        if age == 1:
            deck = self.deck_setup_age_1(self.n_players)
        elif age == 2:
            deck = self.deck_setup_age_2(self.n_players)
        else:
            deck = self.deck_setup_age_3(self.n_players)

        shuffle(deck)
        # Separate the list into lists of 7 elements each
        separated_deck = [deck[i:i+7] for i in range(0, len(deck), 7)]
        i = 0
        for player in self.players:
            player.hand = sorted(separated_deck[i], key=lambda x: x.id)
            #print(f"{player.name}'s hand: {player.hand}")
            i += 1

    def score_game(self):
        reward = [0.0] * self.n_players
        
        for player in self.players:
            for endgame_function in player.endgame_scoring_functions:
                endgame_function(player)
            #player.print_score()
            logger.debug(player.score_string())

        scores = [p.get_total_score() for p in self.players]
        
        if self.alt_reward:
            best_score = max(scores)
            for i, s in enumerate(scores):
                if s == best_score:
                    reward[i] += min(1,(s*2 / sum(scores)))
                else:
                    reward[i] -= min(1,6*(best_score - s)/sum(scores))
        else:
            best_score = max(scores)
            worst_score = min(scores)

            winners = []
            losers = []
            for i, s in enumerate(scores):
                if s == best_score:
                    winners.append(i)
                if s == worst_score:
                    losers.append(i)


            for w in winners:
                #print(w.name,w.get_total_score())
                reward[w] += 1.0 / len(winners)
            
            for l in losers:
                reward[l] -= 1.0 / len(losers)

        if self.log_results:
            self.results_to_file(reward)
        
        #print(reward)
        return reward
    
    def deck_setup_age_3(self,player_count):
        deck = []
        purple_cards = [WorkersGuild(),CraftmensGuild(),TradersGuild(),\
                        PhilosophersGuild(),SpiesGuild(),StrategistsGuild(),\
                        ShipownersGuild(),ScientistsGuild(),MagistratesGuild(),BuildersGuild()] 
        shuffle(purple_cards) #Not all of them are used. Only #player+2

        deck.append(Pantheon())  #7 blue points
        deck.append(Gardens())  #5 blue points
        deck.append(TownHall())  #6 blue points
        deck.append(Palace())  #8 blue points
        deck.append(Senate())  #6 blue points
        deck.append(Haven())  # point + coin per brown card
        deck.append(Lighthouse())  #point + coin per yellow card
        deck.append(Arena())  #3 coins, 1 point per wonder
        deck.append(Fortifications())  #3 shields
        deck.append(Arsenal())  #3 shields
        deck.append(SiegeWorkshop())  #3 shields
        deck.append(Lodge())  #Compass
        deck.append(Observatory())  #Gear
        deck.append(University())  #Tablet
        deck.append(Academy()) #Compass
        deck.append(Study())  #Gear
        deck.append(purple_cards[0])
        deck.append(purple_cards[1])
        deck.append(purple_cards[2])
        deck.append(purple_cards[3])
        deck.append(purple_cards[4])
        if player_count >= 4:
            deck.append(Gardens())  #5 blue points
            deck.append(Haven())  #point + coin per brown card
            deck.append(ChamberOfCommerce())  #2 points + 2 coins per grey card
            deck.append(Circus())  #3 shields
            deck.append(Arsenal())  #3 shields
            deck.append(University()) #Tablet
            deck.append(purple_cards[5])
            if player_count >= 5:
                deck.append(TownHall())  #6 blue points
                deck.append(Senate())  #6 blue points
                deck.append(Arena())  #1 points + 3 coin per wonder
                deck.append(Circus())  #3 shields
                deck.append(SiegeWorkshop())  #3 shields
                deck.append(Study()) #gear
                deck.append(purple_cards[6])
                if player_count >= 6:
                    deck.append(Pantheon()) #7 blue points
                    deck.append(TownHall())  #6 blue points
                    deck.append(Lighthouse())  #point + coin per yellow card
                    deck.append(ChamberOfCommerce())  #2 points + 2 coins per grey card
                    deck.append(Circus())  #3 shields
                    deck.append(Lodge()) # Compass
                    deck.append(purple_cards[7])
                    
        return deck
    def deck_setup_age_2(self,player_count):
        deck = []
        deck.append(Sawmill())  #2 wood
        deck.append(Quarry())  #2 stone
        deck.append(Brickyard())  #2 bricks
        deck.append(Foundry())  #2 ore
        deck.append(Loom())  # loom
        deck.append(Glassworks())  # glass
        deck.append(Press())  #papyrus
        deck.append(Aqueduct())  #5 blue points
        deck.append(Temple())  #3 blue points
        deck.append(Statue())  #4 blue points
        deck.append(Courthouse())  #4 blue points
        deck.append(Forum())  #Free grey resource
        deck.append(Caravansery())  #free brown resource
        deck.append(Vineyard(self.action_bank))  #Gold for brown cards <^>
        deck.append(Walls()) #2 shields
        deck.append(Stables())  #2 shields
        deck.append(ArcheryRange())  #2 shields
        deck.append(Dispensary()) #compass
        deck.append(Laboratory()) #gear
        deck.append(Library())  #tablet
        deck.append(School())  #tablet
        if player_count >= 4:
            deck.append(Sawmill())  #2 wood
            deck.append(Quarry())  #2 stone
            deck.append(Brickyard())  #2 bricks
            deck.append(Foundry())  #2 ore
            deck.append(Bazar(self.action_bank))  #2 Gold for grey cards <^>
            deck.append(TrainingGround()) #2 shields
            deck.append(Dispensary())  #compass
            if player_count >= 5:
                deck.append(Loom())  # loom
                deck.append(Glassworks())  # glass
                deck.append(Press())  #papyrus
                deck.append(Courthouse())  #4 blue points
                deck.append(Caravansery())  #Free brown resource
                deck.append(Stables()) #2 shields
                deck.append(Laboratory())  #gear
                if player_count >= 6:
                    deck.append(Temple()) # 3 blue points
                    deck.append(Forum()) 
                    deck.append(Caravansery())  #Free brown resource
                    deck.append(Vineyard())  #Gold for brown cards <^>
                    deck.append(TrainingGround())  #2 shields
                    deck.append(ArcheryRange()) #2 shields
                    deck.append(Library()) #tablet
                    
        return deck

    def deck_setup_age_1(self,player_count):
        deck = []
        deck.append(LumberYard())  #wood
        deck.append(StonePit())  #stone
        deck.append(ClayPool())  #bricks
        deck.append(OreVein())  #ore
        deck.append(ClayPit())  # ore\bricks
        deck.append(TimberYard())  # wood\stone
        deck.append(Loom())  #glass
        deck.append(Glassworks())  #papyrus
        deck.append(Press())  #loom
        deck.append(Baths())  #3 blue points
        deck.append(Altar())  #2 blue points
        deck.append(Theatre())  #2 blue points
        deck.append(EastTradingPost())  #lower east brown trading costs
        deck.append(WestTradingPost()) #lower west brown trading costs
        deck.append(Marketplace())  #lower both grey trading costs
        deck.append(Stockade())  #lower brown trading costs
        deck.append(Barracks()) 
        deck.append(GuardTower()) 
        deck.append(Apothecary())  #compass
        deck.append(Workshop())  #gear
        deck.append(Scriptorium())  #tablet
        if player_count >= 4:
            deck.append(LumberYard())  #wood
            deck.append(OreVein())  #ore
            deck.append(Excavation())  # stone\bricks
            deck.append(Pawnshop())  #3 blue points
            deck.append(Tavern())  #5 coins
            deck.append(GuardTower())
            deck.append(Scriptorium())  #tablet
            if player_count >= 5:
                deck.append(StonePit())  #stone
                deck.append(ClayPool())  #bricks
                deck.append(ForestCave())  # wood\ore
                deck.append(Altar())  #2 blue points
                deck.append(Tavern())  #5 coins
                deck.append(Barracks())
                deck.append(Apothecary())  #compass
                if player_count >= 6:
                    deck.append(TreeFarm()) # wood\bricks
                    deck.append(Mine())  # stone\ore
                    deck.append(Loom())  #loom
                    deck.append(Glassworks())  #glass
                    deck.append(Press())  #papyrus
                    deck.append(Marketplace())
                    deck.append(Theatre())
                    
        return deck
