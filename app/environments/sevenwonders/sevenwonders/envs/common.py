#!/usr/bin/python
#
# Copyright 2015 - Jonathan Gordon
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
# KIND, either express or implied.

RESOURCE_GOLD 	= "$"
RESOURCE_WOOD 	= "W"
RESOURCE_ORE 	= "O"
RESOURCE_STONE 	= "S"
RESOURCE_BRICK 	= "B"
RESOURCE_GLASS 	= "G"
RESOURCE_LOOM 	= "L"
RESOURCE_PAPYRUS= "P"

#Victory points
POINTS_RED = "VPRed"
POINTS_GOLD = "VPGold"
POINTS_WONDER = "VPWonder"
POINTS_BLUE = "VPBlue"
POINTS_YELLOW = "VPYellow"
POINTS_PURPLE = "VPPurple"
POINTS_GREEN = "VPGreen"


ALL_RESOURCES = {	\
	RESOURCE_GOLD, RESOURCE_WOOD, RESOURCE_ORE,	\
	RESOURCE_STONE, RESOURCE_BRICK, RESOURCE_GLASS,	\
	RESOURCE_LOOM, RESOURCE_PAPYRUS }

BROWN_RESOURCES = {RESOURCE_WOOD, RESOURCE_ORE,RESOURCE_STONE, RESOURCE_BRICK}
GREY_RESOURCES = {RESOURCE_GLASS,RESOURCE_LOOM,RESOURCE_PAPYRUS}


SCIENCE_GEAR 	= "G"
SCIENCE_COMPASS = "C"
SCIENCE_TABLET 	= "T"

ACTION_PLAYCARD = 0
ACTION_DISCARD	= 1
ACTION_STAGEWONDER	= 2

COLOR_BROWN		= "brown"
COLOR_GREY 		= "grey"
COLOR_YELLOW 	= "yellow"
COLOR_GREEN 	= "green"
COLOR_BLUE 		= "blue"
COLOR_RED 		= "red"
COLOR_PURPLE 	= "purple"
COLOR_WONDER = "wonder"

ANSI = {
	COLOR_BROWN : "\033[33m",	
	COLOR_GREY : "\033[37m",
	COLOR_YELLOW : "\033[93m",
	COLOR_GREEN : "\033[92m",
	COLOR_BLUE 	: "\033[94m",
	COLOR_RED : "\033[91m",
	COLOR_PURPLE : "\033[95m",
	COLOR_WONDER : "\033[96m",
	"unavailable" : "\033[90m",
	"default" : "\033[0m"
}


def find_card(cards, name):
	for c in cards:
		if c.get_name() == name:
			return c
	return None
