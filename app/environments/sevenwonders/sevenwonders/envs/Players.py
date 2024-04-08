#!/usr/bin/python
#
# Copyright 2024 - Jakob Cordua Thibodeau
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
# KIND, either express or implied.

from .common import *
from itertools import product
from random import choice


class Player:
	def __init__(self, name, id):
		self.id = id
		self.name = name
		self.under_wonder = [] #for cards put under wonders
		self.tableau = [] # all the players played cards
		self.hand = None
		self.war_losses = 0 # war wins/losses
		self.shields = 0 #Shield/military points
		self.resources = { #This only counts fixed ressources
			RESOURCE_GOLD : 3, #Yes, gold is a resource
			RESOURCE_WOOD 	: 0,
			RESOURCE_ORE 	: 0,
			RESOURCE_STONE 	: 0,
			RESOURCE_BRICK 	: 0,
			RESOURCE_GLASS 	: 0,
			RESOURCE_LOOM 	: 0,
			RESOURCE_PAPYRUS 	: 0
		}
		self.conditional_resources = [] #For brown cards with /
		self.free_conditional_resources = {#For yellow card and wonders with W/B/S/O or G/P/L #Untradable
			COLOR_BROWN:0,
			COLOR_GREY:0
		} 
		self.east_trade_prices = 2 #for brown resources <
		self.west_trade_prices = 2 #for brown resources >
		self.grey_trade_prices = 2 #for both player <>
		self.west_player = None
		self.east_player = None
		self.wonder = None
		self.personality = None
		self.points = {
			POINTS_RED:0,
			POINTS_GOLD:0,
			POINTS_WONDER:0,
			POINTS_BLUE:0,
			POINTS_YELLOW:0,
			POINTS_PURPLE:0,
			POINTS_GREEN:0
		}
		self.science = {
			SCIENCE_GEAR:0,
			SCIENCE_COMPASS:0,
			SCIENCE_TABLET:0,
			"any":0 #for conditional sciences, used at the end when calculating points
		}
		self.color_count = {	#Amount of cards of that color players owns
			COLOR_BROWN:0,
			COLOR_GREY:0,
			COLOR_RED:0,
			COLOR_YELLOW:0,
			COLOR_BLUE:0,
			COLOR_GREEN:0,
			COLOR_PURPLE:0
		}
		#If True, player can play the last card of an age instead of discarding it
		#Still has to pay cost (can also use it to complete wonder or discard for 3 coins)
		self.has_double_last_cards = False 

		self.endgame_scoring_functions = [lambda p: p.score_coins(),lambda p: p.score_science()] #For cards that give points at the end of the game
	
	def set_personality(self, persona):
		self.personality = persona
	
	def set_wonder(self,wonder):
		self.wonder = wonder
	
	def set_east_player(self,east_player):
		self.east_player = east_player
	
	def set_west_player(self,west_player):
		self.west_player = west_player

	def get_name(self):
		return self.name
	
	def get_cards(self):
		return self.tableau
	
	def get_color_count(self,color): 
		return self.color_count[color] #int, number of cards built of that color
	
	def add_resource(self, resource, amount=1):
		self.resources[resource] += amount

	def add_conditional_resource(self,conditional_resource):
		self.conditional_resources.append(conditional_resource)
	
	def add_free_conditional_resource(self,color):
		self.free_conditional_resources[color]+=1 #color = "brown" or "grey"

	def add_points(self,category,amount):
		self.points[category] += amount
		
	def score_coins(self):
		self.points[POINTS_GOLD] += self.resources[RESOURCE_GOLD]//3
	
	def score_science(self):
		scores = []
		anys = self.science["any"]
		if anys == 0:
			self.add_points(POINTS_GREEN,_score_science(self.science))
		elif anys == 1:
			for symbol in [SCIENCE_COMPASS, SCIENCE_TABLET, SCIENCE_GEAR]:
				science = self.science.copy()
				science[symbol] += 1
				scores.append(_score_science(science))
			self.add_points(POINTS_GREEN,max(scores))
		else: #2 anys
			for symbol_1 in [SCIENCE_COMPASS, SCIENCE_TABLET, SCIENCE_GEAR]:
				for symbol_2 in [SCIENCE_COMPASS, SCIENCE_TABLET, SCIENCE_GEAR]:
					science = self.science.copy()
					science[symbol_1] += 1
					science[symbol_2] += 1
					scores.append(_score_science(science))
			self.add_points(POINTS_GREEN,max(scores))

	 
	#For coins only, points are at the end
	#Bazar and Vineyard (that use this) should be put at the end of queue when played	
	def add_coins_per_card(self,amount,card_color,me=True,east=False,west=False,bonus=0):
		if card_color == "wonder":
			self.resources[RESOURCE_GOLD] += amount * self.wonder.stages_completed
			return
		if me:
			self.resources[RESOURCE_GOLD] += amount * self.get_color_count(card_color)
		if east:
			self.resources[RESOURCE_GOLD] += amount * self.east_player.get_color_count(card_color)
		if west:
			self.resources[RESOURCE_GOLD] += amount * self.west_player.get_color_count(card_color)

		self.resources[RESOURCE_GOLD] += bonus * amount  
	
	#point_card_color = card from which points are awarded
	#played_card_color = color/category of card build
	def add_points_per_card(self,amount,played_card_color,point_card_color,me=True,east=False,west=False):
		if me:
			self.points[played_card_color] += amount * self.get_color_count(point_card_color) 
		if east:
			self.points[played_card_color] += amount * self.east_player.get_color_count(point_card_color)
		if west:
			self.points[played_card_color] += amount * self.west_player.get_color_count(point_card_color)

	def add_points_per_wonder(self,amount,played_card_color,me=True,east=False,west=False):
		if me:
			self.points[played_card_color] += amount * self.wonder.stages_completed
		if east:
			self.points[played_card_color] += amount * self.east_player.wonder.stages_completed
		if west:
			self.points[played_card_color] += amount * self.west_player.wonder.stages_completed

	def lower_trading_cost(self,trade_type):
		if trade_type == "east": #east brown cards
			self.east_trade_prices = 1
		elif trade_type == "west": #west brown cards
			self.west_trade_prices = 1
		else: #if "grey"
			self.grey_trade_prices = 1

	def add_shields(self, amount): #shields for war
		self.shields += amount
	
	def add_science(self, symbol):
		self.science[symbol]+=1 #"any" for a conditional science card/wonder"

	def can_double_last_cards(self):
		self.has_double_last_cards = True

	def get_total_score(self):
		return sum(self.points.values())
	
	def resources_as_list(self):
		resources_list = []
		for resource, amount in self.resources.items():
			for _ in range(amount):
				resources_list.append(resource)
		return resources_list


	def score_string(self):
		return f"{self.name}'s score breakdown:\n"\
		f"-------------------------------\n"\
		f"{ANSI[COLOR_RED]}War Points: {self.points[POINTS_RED]} pts\n"\
		f"{ANSI[COLOR_BROWN]}Gold Points: {self.points[POINTS_GOLD]} pts\n"\
		f"{ANSI[COLOR_WONDER]}Wonder Points: {self.points[POINTS_WONDER]} pts\n"\
		f"{ANSI[COLOR_BLUE]}Blue Points: {self.points[POINTS_BLUE]} pts\n"\
		f"{ANSI[COLOR_YELLOW]}Yellow Points: {self.points[POINTS_YELLOW]} pts\n"\
		f"{ANSI[COLOR_PURPLE]}Purple Points: {self.points[POINTS_PURPLE]} pts\n"\
		f"{ANSI[COLOR_GREEN]}Green Points: {self.points[POINTS_GREEN]} pts\n"\
		f"{ANSI['default']}-------------------------------\n"\
		f"Total: {self.get_total_score()} pts\n"

	def print_score(self):
		print(self.score_string())
		

	
	def print_hand(self):
		print(self.hand)	

	def print_tableau(self):
		print(self.tableau)

	def war(self,points_given):
		if self.west_player.shields > self.shields:
			#print(f"{self.name} loses against {self.west_player.name}")
			self.war_losses += 1
			self.points[POINTS_RED] -= 1
		elif self.west_player.shields < self.shields:
			#print(f"{self.name} wins against {self.west_player.name}")
			self.points[POINTS_RED] += points_given
		
			
			#print(f"{self.name} ties against {self.west_player.name}")



		if self.east_player.shields > self.shields:
			self.war_losses += 1
			self.points[POINTS_RED] -= 1
			#print(f"{self.name} loses against {self.east_player.name}")
		elif self.east_player.shields < self.shields:
			#print(f"{self.name} wins against {self.east_player.name}")
			self.points[POINTS_RED] += points_given
		
			
			#print(f"{self.name} ties against {self.east_player.name}")
	def print_available_cards(self, hand_cost):
		hand = self.hand.copy()
		for i in range(len(hand)):
			price = hand_cost[i]
			card = hand[i]
			if price == -1:
				print(f"{ANSI['unavailable']}[{card.id+1}] play ".ljust(18)+f"{card.name}\033[0m") 
				

				print(f"[{card.id+76}] discard ".ljust(13)+f"{str(card)}")
				
				continue
				

			elif price == 0:
				p = ""
			elif price == 1:
				p = "Bank: $"
			else:
				
				p = ("East: "+"$"*price['east'] if price['east'] > 0 else "")\
				+("     " if price['east']>0 and price['west']>0 else "")\
				+("West: "+"$"*price['west'] if price['west'] > 0 else "")

			
			print(f"[{card.id+1}] play ".ljust(13)+f"{str(card).ljust(26)} {p}")
			

			print(f"[{card.id+76}] discard".ljust(13)+f"{str(card)}")
			
					
	def print_wonder_option(self,price,wonder_available):
		if not wonder_available:
			print(f"{ANSI['unavailable']}[0] play ".ljust(18)+f"{self.wonder.name.ljust(26)}\033[0m")
		else:
			if price == 0:
				p = ""
			elif price == 1:
				p = "Bank: $"
			else:
				p = ("East: "+"$"*price['east'] if price['east'] > 0 else "")\
				+("     " if price['east']>0 and price['west']>0 else "")\
				+("West: "+"$"*price['west'] if price['west'] > 0 else "")
			
			print(f"[0] play ".ljust(13)+f"{str(self.wonder).ljust(26)} {p}")
	
		
	def choose_card_for_wonder(self,input_option):
		#print("You've selected your wonder, please choose a card to play under your wonder: ")
		#for card in self.hand:
			#print(f"[{input_option}] {str(card)}")
			#input_option += 1
		card = self.hand.pop(input_option//2 - 1)
		self.under_wonder.append(card)
		return f'\n{self.name} tucked {card} under their wonder'

	def play_from_discard(self,input,discard_pile):
		if not discard_pile: #empty discard pile
			#print(f"{self.name}, there are no cards in the discard pile\n")
			return
		else:
			#print(f"{self.name}, You can play a card from the discard pile for free:")
			#input_option = 1
			for i,card in enumerate(discard_pile):
				if card.id == input-1:
					to_play = discard_pile.pop(i)

				#print(f"[{input_option}] {str(card)}")
				 
			self.play_card(to_play,0) #free card
			return to_play
			
	def play_card(self,card,cost):
		if cost != 0 and cost != 1: #Pay the players you bought shit from
			self.east_player.resources[RESOURCE_GOLD] += cost['east']
			self.west_player.resources[RESOURCE_GOLD] += cost['west']

		self.color_count[card.color] += 1
		self.tableau.append(card)
		card.effect(self)
		
	
	def play_wonder(self,cost):
		if cost != 0: #Pay the players you bought shit from
			self.east_player.resources[RESOURCE_GOLD] += cost['east']
			self.west_player.resources[RESOURCE_GOLD] += cost['west']

		self.wonder.effect(self)
	
	def get_hand_cost(self):
		cost = []
		for card in self.hand:
			cost.append(self.get_price(card))
		return cost

#returns the cost in gold of buying this card
	#if 0, the card is free
	#if 1, card costs 1 TO THE BANK
	#if -1 the card cannot be bought (won't be available)
	#if [w,e] w is amount paid to west player and e to east player

	def get_price(self,card): #this can also be a Wonder
		
		if self.is_free_prechains(card):
			return 0

		cost = card.get_cost()
	
		#If the card is free, price is 0
		if cost is None:
			return 0
		
		#If the card costs 1 gold, can either afford it (1 gold) or not (-1)
		if cost[0] == RESOURCE_GOLD:
			if self.resources[RESOURCE_GOLD]>0:
				return 1
			return -1
		
    	# Create a copy of the player's resources dictionary
		available_resources = self.resources.copy()
		

		#Seperate into brown and grey (easier for conditional resources)
		new_cost_brown = []
		new_cost_grey = []
		#Go through set resources
		for resource in cost:
			if available_resources[resource] > 0:
				available_resources[resource] -= 1
			else:
				new_cost_brown.append(resource) if resource in BROWN_RESOURCES else new_cost_grey.append(resource)
		
		#If could be paid with basic resources and/or yellow conditional resources
		if max(len(new_cost_brown)-self.free_conditional_resources[COLOR_BROWN],0)\
		  + max(len(new_cost_grey)-self.free_conditional_resources[COLOR_GREY],0) == 0:
			return 0

		possible_prices = []
		#Buying from other people and/or using conditional resources
		if len(new_cost_brown)-self.free_conditional_resources[COLOR_BROWN] <= 0: #Only grey resources left and have to buy some
			return self.buy_grey_from_neighbors(new_cost_grey)
		
		elif len(new_cost_grey)-self.free_conditional_resources[COLOR_GREY] <= 0: #Only brown resources left 
			for combo in list(product(*self.conditional_resources)):
				combo = list(combo) 
				new_new_cost_brown = []
				for resource in new_cost_brown:
					if resource in combo:
						combo.remove(resource) 
					else:
						new_new_cost_brown.append(resource)
						
				if len(new_new_cost_brown) - self.free_conditional_resources[COLOR_BROWN] < 1:
					return 0
				else:
					price = self.buy_brown_from_neighbors(new_new_cost_brown)
					if price == {'east':self.east_trade_prices,'west':0} or price == {'east':0,'west':self.west_trade_prices}: #it doesn't get better than this
						return price
					elif price != -1:
						possible_prices.append(price)
		else: #need to buy brown and grey
			#Here we know there is at least one grey resource missing
			grey_price = self.buy_grey_from_neighbors(new_cost_grey)
			if grey_price == -1: #if you can't buy grey resources from neighbors, then you can't buy the card
				return -1
			


			#Generates all combinations of conditional resources
			for combo in list(product(*self.conditional_resources)): 
				combo = list(combo)
				#For each combination, check if can be paid with those resources
				new_new_cost_brown = []
				for resource in new_cost_brown:
					if resource in combo:
						combo.remove(resource) 
					else:
						new_new_cost_brown.append(resource)

				#If could be brown paid with basic resources and/or yellow conditional resources
				if len(new_new_cost_brown) - self.free_conditional_resources[COLOR_BROWN] < 1:
					return grey_price
				else:
					brown_price = self.buy_brown_from_neighbors(new_new_cost_brown) 
					if brown_price != -1:
						possible_prices.append({'east':brown_price["east"]+grey_price["east"],'west':brown_price["west"]+grey_price["west"]}) #Combine grey and brown prices
		
		min_price = find_min_price(possible_prices)
		if min_price == -1 or min_price['east']+min_price['west'] > self.resources[RESOURCE_GOLD]:
			return -1
		else:
			return min_price 
	

	def get_player_log(self,win):
		self.points["total"] = sum(self.points.values())
		log = {"name": self.name,
		 	   "win": win,
               "tableau":[card.id for card in self.tableau],
			   "color_count":self.color_count,
			   "points":self.points,
			   "wonder":self.wonder.id,
			   }
		return log 

	def is_free_prechains(self,card):
		if not hasattr(card,"prechains"):
			return False
		for pre in card.prechains:
			for card in self.tableau:
				if card.name == pre:
					return True
		return False
	
	#Even for wonders that need 2 grey ressources, you'll never need to buy more than one per type
	def buy_grey_from_neighbors(self,grey_cost):
		both = 0
		east_cost = 0
		west_cost = 0
		free_grey = self.free_conditional_resources[COLOR_GREY]

		east_resources = self.east_player.resources.copy()
		west_resources = self.west_player.resources.copy()
		for resource in grey_cost:
			if east_resources[resource]>0 and west_resources[resource]>0:
				both += 1 #This is "amount of resources both have" not price
			elif east_resources[resource]>0:
				east_cost += self.grey_trade_prices
			elif west_resources[resource]>0:
				west_cost += self.grey_trade_prices
			else:#if none of them have it
				if free_grey > 0: #"Use" a free conditional free resource if you have any left
					free_grey -= 1
				else: #If nobody has the resource and you don't have any free ones, you can't buy the card
					return -1
		
		#Time to use those free conditional resources if you haven't already
		#Here you are guaranteed to have something to pay
		both,west_cost,east_cost = self.assign_free_grey(free_grey,both,west_cost,east_cost)

		cost = {"east":east_cost,"west":west_cost}

		if both > 1: #if 2 or 3
			cost["west"] += self.grey_trade_prices 
			cost["east"] += self.grey_trade_prices
		if both%2 == 1: #if 1 or 3
			#Give the player with the least coins more coins OR
			#Give a random player more coins 
			#This is a compromise for AI, could be changed later for players
			#FIXME
			if cost["east"] > cost['west']:
				cost["west"] += self.grey_trade_prices
			elif cost["east"] < cost['west']:
				cost["west"] += self.grey_trade_prices
			else:
				cost[choice(["east","west"])] += self.grey_trade_prices

		return cost if sum(cost.values()) <= self.resources[RESOURCE_GOLD] else -1

	def assign_free_grey(self,free_grey,both,west_cost,east_cost):
		#FIXME players should be able to choose
		#Made to avoid the "both" logic later
		#But also made to distribute equally (if [1,2,0] it will transform to [1,1,0])
		for _ in range(free_grey):
			if both == west_cost or both == east_cost: #[1,1,1] or [1,1,0] or [1,0,1]
				both-=1
			elif west_cost == east_cost: #[0,1,1]
				if choice([True,False]):
					west_cost -= self.grey_trade_prices
				else:
					east_cost -= self.grey_trade_prices
			else:
				max_value = max(both,west_cost,east_cost)
				if both == max_value: 
					both -= 1
				elif west_cost == max_value:
					west_cost -= self.grey_trade_prices
				else:
					east_cost -= self.grey_trade_prices

		return both,west_cost,east_cost
	
	def buy_brown_from_neighbors(self,brown_cost):
		prices = []

		if self.east_trade_prices == self.west_trade_prices: #If both same price	
			prices = self.get_prices_neighbors_same(brown_cost)

		elif self.east_trade_prices == 1 and self.west_trade_prices == 2: #If east player is cheaper
			prices = self.get_prices_neighbors_east_cheaper(brown_cost)
		
		else:	#If west player is cheaper
			prices = self.get_prices_neighbors_west_cheaper(brown_cost)
		
		return find_min_price(prices)

	def get_prices_neighbors_east_cheaper(self,brown_cost):
		prices = []
		east_trade_price = self.east_trade_prices
		west_trade_price = self.west_trade_prices

		for east_combo in list(product(*self.east_player.conditional_resources)):
			for west_combo in list(product(*self.west_player.conditional_resources)):
				east_cost = 0
				west_cost = 0
				free_brown = self.free_conditional_resources[COLOR_BROWN]
				
				#Get neighbor resources
				east_resources = self.east_player.resources.copy()
				west_resources = self.west_player.resources.copy()
				#Add resources from combo
				for resource in east_combo:
					east_resources[resource] += 1
				for resource in west_combo:
					west_resources[resource] += 1
				have_resources = True
				#Now we check if we can afford it
				for resource in brown_cost:
					if east_resources[resource]>0: #if east player has it
						east_cost += east_trade_price
						east_resources[resource] -= 1
					elif west_resources[resource]>0: #if west player has it
						west_cost += west_trade_price
						west_resources[resource] -= 1
					else:#if none of them have it 
						if free_brown>0: #Use a free conditional free resource if you have any left
							free_brown -= 1
						else:	#Cannot buy the card
							have_resources = False
							break
				
				if not(have_resources and \
					(sum([max(east_cost-free_brown*1,0),west_cost])<=self.resources[RESOURCE_GOLD] or\
					sum([east_cost,max(west_cost-free_brown*2,0)])<=self.resources[RESOURCE_GOLD])):
					continue

				#use free browns if you have some left
				for _ in range(free_brown):
					if east_cost > 0:
						east_cost -= east_trade_price
					else:
						west_cost -= west_trade_price
				
				prices.append({"east":east_cost,"west":west_cost})

		return prices

	def get_prices_neighbors_west_cheaper(self,brown_cost):
		prices = []
		east_trade_price = self.east_trade_prices
		west_trade_price = self.west_trade_prices
		
		for east_combo in list(product(*self.east_player.conditional_resources)):
			for west_combo in list(product(*self.west_player.conditional_resources)):
				east_cost = 0
				west_cost = 0
				free_brown = self.free_conditional_resources[COLOR_BROWN]
				
				#Get neighbor resources
				east_resources = self.east_player.resources.copy()
				west_resources = self.west_player.resources.copy()
				#Add resources from combo
				for resource in east_combo:
					east_resources[resource] += 1
				for resource in west_combo:
					west_resources[resource] += 1
				have_resources = True
				#Now we check if we can afford it
				for resource in brown_cost:
					if west_resources[resource]>0: #if east player has it
						west_cost += west_trade_price
						west_resources[resource] -= 1
					elif east_resources[resource]>0: #if west player has it
						east_cost += east_trade_price
						east_resources[resource] -= 1
					else:#if none of them have it 
						if free_brown>0: #Use a free conditional free resource if you have any left
							free_brown -= 1
						else:	#Cannot buy the card
							have_resources = False
							break
				
				if not(have_resources and \
					(sum([max(east_cost-free_brown*2,0),west_cost])<=self.resources[RESOURCE_GOLD] or\
					sum([east_cost,max(west_cost-free_brown*1,0)])<=self.resources[RESOURCE_GOLD])):
					continue

				#use free browns if you have some left
				for _ in range(free_brown):
					if west_cost > 0:
						west_cost -= west_trade_price
					else:
						east_cost -= east_trade_price
				
				prices.append({"east":east_cost,"west":west_cost})

		return prices

	def get_prices_neighbors_same(self,brown_cost):
		trade_price = self.east_trade_prices
		prices = []
		#Create alternate universes for each conditional resource
		for east_combo in list(product(*self.east_player.conditional_resources)):
			for west_combo in list(product(*self.west_player.conditional_resources)):
				
				both = []
				east_cost = 0
				west_cost = 0
				free_brown = self.free_conditional_resources[COLOR_BROWN]
				
				#Get neighbor resources
				east_resources = self.east_player.resources.copy()
				west_resources = self.west_player.resources.copy()
				#Add resources from combo
				for resource in east_combo:
					east_resources[resource] += 1
				for resource in west_combo:
					west_resources[resource] += 1
				have_resources = True
				#Now we check if we can afford it
				for resource in brown_cost:
					if east_resources[resource]>0 and west_resources[resource]>0:
						both.append(resource) #Add this resource to both
						east_resources[resource] -= 1
						west_resources[resource] -= 1
					elif east_resources[resource]>0: #if east player has it
						east_cost += trade_price
						east_resources[resource] -= 1
					elif west_resources[resource]>0: #if west player has it
						west_cost += trade_price
						west_resources[resource] -= 1
					else:#if none of them have it 
						if resource in both: #If it was previously added to both
							both.remove(resource)
							east_cost += trade_price
							west_cost += trade_price
						elif free_brown>0: #Use a free conditional free resource if you have any left
							free_brown -= 1
						else:	#Cannot buy the card
							have_resources = False
							break
				
				both = len(both) #Amount of unused resources owned by both players
				#If you cannot afford the card with this combination of resources
				if not(have_resources and sum([east_cost,west_cost,both*trade_price])-free_brown*trade_price <= self.resources[RESOURCE_GOLD]):
					continue
				
				#Use free browns you have left
				for _ in range(free_brown):
					if east_cost > west_cost: #Use free brown on east 
						east_cost -= trade_price
					elif east_cost < west_cost: #Use free brown on west 
						west_cost -= trade_price
					elif both > 0: #Use free brown on both 
						both -= 1
					else: #if no both and equal payout, just choose one randomly
						if choice([True,False]):
							east_cost -= trade_price
						else:
							west_cost -= trade_price
			
				cost = {"east":east_cost,"west":west_cost}

				if both > 1: #if 2,3,4,5,6 give them both the same price for pair number
					cost["west"] += trade_price*(both//2)
					cost["east"] += trade_price*(both//2)
				if both%2 == 1: #if 1,3,5
					#Give the player with the least coins more coins OR
					#Give a random player more coins 
					#This is a compromise for AI, could be changed later for players
					#FIXME
					if cost["east"] > cost['west']:
						cost["west"] += trade_price
					elif cost["east"] < cost['west']:
						cost["west"] += trade_price
					else:
						cost[choice(["east","west"])] += trade_price

				prices.append(cost)
				
		return prices


def find_min_price(prices):

	if len(prices) == 0:
		return -1
	
	min_price = {"east":1000,"west":1000}
	sum_price = 2000
	
	for price in prices:
		if sum(price.values()) < sum_price:
			min_price = price
			sum_price = sum(price.values())

	return min_price

def _score_science(science):
	score = 0				
	score += science[SCIENCE_GEAR]**2
	score += science[SCIENCE_COMPASS]**2
	score += science[SCIENCE_TABLET]**2
	score += min([science[SCIENCE_TABLET],science[SCIENCE_GEAR],science[SCIENCE_COMPASS]])*7
	return score