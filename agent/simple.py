"""
Simple Agent Module

Provides a streamlined approach for direct frame + state -> action processing,
with enhanced history tracking to prevent getting stuck in loops.

Key improvements over the original simple mode:
- Location-based stuck detection (tracks repeated actions at same coordinates)
- Context-aware history (overworld/battle/menu/dialogue awareness)  
- Memory management to fit within LLM context limits
- Detailed history tracking with timestamps and game state summaries
- Smart context switching that helps agent avoid infinite loops
- Configurable history window sizes for different use cases
- Chain of thought reasoning with structured LLM responses
- Objectives system with automatic and manual completion tracking
- Dynamic goal setting and progress monitoring

The agent maintains objectives (go to location, battle trainer, etc.) that are
automatically tracked and marked complete when achieved. The LLM can also
manually complete objectives and create new ones dynamically through structured
commands. It uses chain of thought reasoning to make better decisions while
considering current objectives. All state including objectives is forwarded
to support external monitoring and debugging.

Configuration defaults (can be customized):
- 100 previous state/location entries (with context and reasoning)
- 50 recent button presses tracked  
- 15 history entries shown to LLM in prompts
- 20 recent actions shown to LLM in prompts
- Automatic memory management to stay within LLM context limits
"""

import logging
import os
import sys
import time
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import heapq

from utils.state_formatter import (
    format_state_for_llm,
    format_movement_preview_for_llm,
    _format_map_info,
)

logger = logging.getLogger(__name__)

# Configurable parameters for history tracking
DEFAULT_MAX_HISTORY_ENTRIES = 100  # Previous states/locations with context
DEFAULT_MAX_RECENT_ACTIONS = 50    # Recent button presses
DEFAULT_HISTORY_DISPLAY_COUNT = 30 # Number of history entries shown to LLM
DEFAULT_ACTIONS_DISPLAY_COUNT = 5 # Number of recent actions shown to LLM
DEFAULT_MOVEMENT_MEMORY_CLEAR_INTERVAL = 30  # Clear movement memory after N actions (0 = never clear)

def configure_simple_agent_defaults(max_history_entries: int = None, max_recent_actions: int = None, 
                                  history_display_count: int = None, actions_display_count: int = None,
                                  movement_memory_clear_interval: int = None):
    """Configure default parameters for all new SimpleAgent instances"""
    global DEFAULT_MAX_HISTORY_ENTRIES, DEFAULT_MAX_RECENT_ACTIONS
    global DEFAULT_HISTORY_DISPLAY_COUNT, DEFAULT_ACTIONS_DISPLAY_COUNT
    global DEFAULT_MOVEMENT_MEMORY_CLEAR_INTERVAL
    
    if max_history_entries is not None:
        DEFAULT_MAX_HISTORY_ENTRIES = max_history_entries
    if max_recent_actions is not None:
        DEFAULT_MAX_RECENT_ACTIONS = max_recent_actions
    if history_display_count is not None:
        DEFAULT_HISTORY_DISPLAY_COUNT = history_display_count
    if actions_display_count is not None:
        DEFAULT_ACTIONS_DISPLAY_COUNT = actions_display_count
    if movement_memory_clear_interval is not None:
        DEFAULT_MOVEMENT_MEMORY_CLEAR_INTERVAL = movement_memory_clear_interval
        
    logger.info(f"Updated SimpleAgent defaults: {DEFAULT_MAX_HISTORY_ENTRIES} history, {DEFAULT_MAX_RECENT_ACTIONS} actions, "
               f"display {DEFAULT_HISTORY_DISPLAY_COUNT}/{DEFAULT_ACTIONS_DISPLAY_COUNT}, "
               f"movement memory clear interval: {DEFAULT_MOVEMENT_MEMORY_CLEAR_INTERVAL}")

@dataclass
class Objective:
    """Single objective/goal for the agent"""
    id: str
    description: str
    objective_type: str  # "location", "battle", "item", "dialogue", "custom"
    target_value: Optional[Any] = None  # Specific target (coords, trainer name, item name, etc.)
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    progress_notes: str = ""
    storyline: bool = False  # True for main storyline objectives (auto-verified), False for agent sub-objectives
    milestone_id: Optional[str] = None  # Emulator milestone ID for storyline objectives

@dataclass
class HistoryEntry:
    """Single entry in the agent's history"""
    timestamp: datetime
    player_coords: Optional[Tuple[int, int]]
    map_id: Optional[int]
    context: str  # "overworld", "battle", "menu", "dialogue"
    action_taken: str
    game_state_summary: str

@dataclass
class SimpleAgentState:
    """Maintains history and state for the simple agent"""
    # Note: We don't use defaults here because they're captured at class definition time
    history: deque = None
    recent_actions: deque = None
    stuck_detection: Dict[str, int] = field(default_factory=dict)
    step_counter: int = 0
    objectives: List[Objective] = field(default_factory=list)
    objectives_updated: bool = False
    failed_movements: Dict[str, List[str]] = field(default_factory=dict)  # coord_key -> [failed_directions]
    npc_interactions: Dict[str, str] = field(default_factory=dict)  # coord_key -> interaction_notes
    movement_memory_action_counter: int = 0  # Counter for tracking actions since last memory clear

    def __post_init__(self):
        """Initialize deques with current default values"""
        if self.history is None:
            self.history = deque(maxlen=DEFAULT_MAX_HISTORY_ENTRIES)
        if self.recent_actions is None:
            self.recent_actions = deque(maxlen=DEFAULT_MAX_RECENT_ACTIONS)

class SimpleAgent:
    """
    Simple agent that processes frame + state -> action directly with history tracking
    """
    
    def __init__(self, vlm, max_history_entries: int = None, max_recent_actions: int = None, 
                 history_display_count: int = None, actions_display_count: int = None,
                 movement_memory_clear_interval: int = None):
        self.vlm = vlm
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_log_dir = os.path.join("frame_logs", session_id)
        os.makedirs(self.frame_log_dir, exist_ok=True)
        self._last_logged_frame_hash: Optional[str] = None
        
        # Use current global defaults if not specified
        max_history_entries = max_history_entries or DEFAULT_MAX_HISTORY_ENTRIES
        max_recent_actions = max_recent_actions or DEFAULT_MAX_RECENT_ACTIONS
        history_display_count = history_display_count or DEFAULT_HISTORY_DISPLAY_COUNT
        actions_display_count = actions_display_count or DEFAULT_ACTIONS_DISPLAY_COUNT
        movement_memory_clear_interval = movement_memory_clear_interval if movement_memory_clear_interval is not None else DEFAULT_MOVEMENT_MEMORY_CLEAR_INTERVAL
        
        self.state = SimpleAgentState()
        self.state.history = deque(maxlen=max_history_entries)
        self.state.recent_actions = deque(maxlen=max_recent_actions)

        self.story_objective_completed = False

        self.reasoning_effort: Optional[str] = "medium"
        
        # Display parameters for LLM prompts
        self.history_display_count = history_display_count
        self.actions_display_count = actions_display_count
        
        # Movement memory clearing interval
        self.movement_memory_clear_interval = movement_memory_clear_interval
        
        # Initialize storyline objectives for Emerald progression
        self._initialize_storyline_objectives()

        self.prev_analysis = []

        self.memories = []

        self.response_history = deque(maxlen=50)

        self.last_turn_actions = []

        self.interact_destination_list = []

    def _complete_all_added_objectives(self, reason: str = "Reset before planner update"):
        """Mark all non-storyline objectives as completed (used when refreshing planner guidance)."""
        any_completed = False
        for obj in self.state.objectives:
            if not obj.storyline and not obj.completed:
                obj.completed = True
                obj.completed_at = datetime.now()
                obj.progress_notes = reason
                any_completed = True
        if any_completed:
            self.state.objectives_updated = True

    def set_reasoning_effort(self, effort: Optional[str]):
        """Adjust reasoning effort for subsequent model calls."""
        valid_values = {None, "low", "medium", "high"}
        normalized = effort.lower() if isinstance(effort, str) else effort
        if normalized not in valid_values:
            logger.warning(f"Ignoring unsupported reasoning effort '{effort}'. Valid options: low, medium, high, or None.")
            return
        self.reasoning_effort = normalized
        
    def _initialize_storyline_objectives(self):
        """Initialize the main storyline objectives for Pokémon Emerald progression"""
        storyline_objectives = [
            {
                "id": "story_game_start",
                "description": "Complete title sequence and begin the game",
                "objective_type": "system",
                "target_value": "Game Running",
                "milestone_id": "GAME_RUNNING"
            },
            {
                "id": "story_intro_complete",
                "description": "Complete intro cutscene with moving van",
                "objective_type": "cutscene",
                "target_value": "Intro Complete",
                "milestone_id": "INTRO_CUTSCENE_COMPLETE"
            },
            {
                "id": "story_player_house",
                "description": "Enter player's house for the first time",
                "objective_type": "location",
                "target_value": "Player's House",
                "milestone_id": "PLAYER_HOUSE_ENTERED"
            },
            {
                "id": "story_player_bedroom",
                "description": "Go upstairs to player's bedroom",
                "objective_type": "location",
                "target_value": "Player's Bedroom",
                "milestone_id": "PLAYER_BEDROOM"
            },
            {
                "id": "story_rival_house",
                "description": "Visit May's house next door",
                "objective_type": "location",
                "target_value": "Rival's House",
                "milestone_id": "RIVAL_HOUSE"
            },
            {
                "id": "story_rival_bedroom",
                "description": "Visit May's bedroom on the second floor",
                "objective_type": "location",
                "target_value": "Rival's Bedroom",
                "milestone_id": "RIVAL_BEDROOM"
            },
            {
                "id": "story_route_101",
                "description": "Travel north to Route 101 and encounter Prof. Birch",
                "objective_type": "location",
                "target_value": "Route 101",
                "milestone_id": "ROUTE_101"
            },
            {
                "id": "story_starter_chosen",
                "description": "Choose starter Pokémon and receive first party member",
                "objective_type": "pokemon",
                "target_value": "Starter Pokémon",
                "milestone_id": "STARTER_CHOSEN"
            },
            {
                "id": "story_birch_lab",
                "description": "Visit Professor Birch's lab in Littleroot Town and receive the Pokedex",
                "objective_type": "location",
                "target_value": "Birch's Lab",
                "milestone_id": "BIRCH_LAB_VISITED"
            },
            {
                "id": "story_oldale_town",
                "description": "Leave lab and continue journey north to Oldale Town",
                "objective_type": "location",
                "target_value": "Oldale Town",
                "milestone_id": "OLDALE_TOWN"
            },
            {
                "id": "story_route_103",
                "description": "Travel to Route 103 to meet rival. You MUST talk to your rival here!!",
                "objective_type": "location",
                "target_value": "Route 103",
                "milestone_id": "ROUTE_103"
            },
            {
                "id": "story_received_pokedex",
                "description": "Return to Birch's lab AND talk to Professor Birch to receive the Pokédex",
                "objective_type": "item",
                "target_value": "Pokédex",
                "milestone_id": "RECEIVED_POKEDEX"
            },
            {
                "id": "story_route_102",
                "description": "Return through Route 102 toward Petalburg City",
                "objective_type": "location",
                "target_value": "Route 102",
                "milestone_id": "ROUTE_102"
            },
            {
                "id": "story_petalburg_city",
                "description": "Navigate to Petalburg City and visit Dad's gym",
                "objective_type": "location",
                "target_value": "Petalburg City",
                "milestone_id": "PETALBURG_CITY"
            },
            {
                "id": "story_dad_meeting",
                "description": "Meet Dad at Petalburg City Gym",
                "objective_type": "dialogue",
                "target_value": "Dad Meeting",
                "milestone_id": "DAD_FIRST_MEETING"
            },
            {
                "id": "story_gym_explanation",
                "description": "Receive explanation about Gym challenges",
                "objective_type": "dialogue",
                "target_value": "Gym Tutorial",
                "milestone_id": "GYM_EXPLANATION"
            },
            {
                "id": "story_route_104_south",
                "description": "Travel through southern section of Route 104",
                "objective_type": "location",
                "target_value": "Route 104 South",
                "milestone_id": "ROUTE_104_SOUTH"
            },
            {
                "id": "story_petalburg_woods",
                "description": "Navigate through Petalburg Woods to help Devon researcher",
                "objective_type": "location",
                "target_value": "Petalburg Woods",
                "milestone_id": "PETALBURG_WOODS"
            },
            {
                "id": "story_aqua_grunt",
                "description": "Defeat Team Aqua Grunt in Petalburg Woods",
                "objective_type": "battle",
                "target_value": "Aqua Grunt Defeated",
                "milestone_id": "TEAM_AQUA_GRUNT_DEFEATED"
            },
            {
                "id": "story_route_104_north",
                "description": "Travel through northern section of Route 104 to Rustboro",
                "objective_type": "location",
                "target_value": "Route 104 North",
                "milestone_id": "ROUTE_104_NORTH"
            },
            {
                "id": "story_rustboro_city",
                "description": "Arrive in Rustboro City and deliver Devon Goods",
                "objective_type": "location",
                "target_value": "Rustboro City",
                "milestone_id": "RUSTBORO_CITY"
            },
            {
                "id": "story_rustboro_gym",
                "description": "Enter the Rustboro Gym and challenge Roxanne",
                "objective_type": "location",
                "target_value": "Rustboro Gym",
                "milestone_id": "RUSTBORO_GYM_ENTERED"
            },
            {
                "id": "story_roxanne_defeated",
                "description": "Defeat Gym Leader Roxanne",
                "objective_type": "battle",
                "target_value": "Roxanne Defeated",
                "milestone_id": "ROXANNE_DEFEATED"
            },
            {
                "id": "story_stone_badge",
                "description": "Receive the Stone Badge and complete first gym",
                "objective_type": "badge",
                "target_value": "Stone Badge",
                "milestone_id": "FIRST_GYM_COMPLETE"
            }
        ]
        
        # Add storyline objectives to the state
        for obj_data in storyline_objectives:
            objective = Objective(
                id=obj_data["id"],
                description=obj_data["description"],
                objective_type=obj_data["objective_type"],
                target_value=obj_data["target_value"],
                completed=False,
                progress_notes="Storyline objective - verified by emulator milestones",
                storyline=True,
                milestone_id=obj_data["milestone_id"]
            )
            self.state.objectives.append(objective)

        logger.info(f"Initialized {len(storyline_objectives)} storyline objectives for Emerald progression (up to first gym)")
        
    def get_game_context(self, game_state: Dict[str, Any]) -> str:
        """Determine current game context (overworld, battle, menu, dialogue)"""
        try:
            # Check if in title sequence first
            player_location = game_state.get("player", {}).get("location", "")
            if player_location == "TITLE_SEQUENCE":
                return "title"
            
            # Check game state for title/intro
            game_state_value = game_state.get("game", {}).get("game_state", "").lower()
            if "title" in game_state_value or "intro" in game_state_value:
                return "title"
            
            # Check if player name is not set (indicates title sequence)
            player_name = game_state.get("player", {}).get("name", "").strip()
            if not player_name or player_name == "????????":
                return "title"
            
            # Check if in battle
            is_in_battle = game_state.get("game", {}).get("is_in_battle", False)
            if is_in_battle:
                logger.debug(f"Detected battle context")
                return "battle"
            
            # Check if dialogue is active
            dialogue_state = game_state.get("game", {}).get("dialogue", {})
            if dialogue_state.get("active", False) or dialogue_state.get("text", "").strip():
                return "dialogue"
            
            # Check if in menu (simplified detection)
            # Could be enhanced with more sophisticated menu detection
            player_state = game_state.get("player", {})
            if player_state.get("in_menu", False):
                return "menu"
            
            # Default to overworld
            return "overworld"
            
        except Exception as e:
            logger.warning(f"Error determining game context: {e}")
            return "unknown"
    
    def get_player_coords(self, game_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract player coordinates from game state"""
        try:
            player = game_state.get("player", {})
            # Try position.x/y first (standard format)
            position = player.get("position", {})
            if position:
                x = position.get("x")
                y = position.get("y")
                if x is not None and y is not None:
                    return (x, y)
            
            # Fallback: try direct x/y on player
            x = player.get("x")
            y = player.get("y")
            if x is not None and y is not None:
                return (x, y)
        except Exception as e:
            logger.warning(f"Error getting player coords: {e}")
        return None
    
    def get_map_id(self, game_state: Dict[str, Any]) -> Optional[int]:
        """Extract map ID from game state"""
        try:
            return game_state.get("map", {}).get("id")
        except Exception as e:
            logger.warning(f"Error getting map ID: {e}")
        return None
    
    def add_objective(self, description: str, objective_type: str, target_value: Any = None) -> str:
        """Add a new objective and return its ID"""
        obj_id = f"obj{len(self.state.objectives)}"
        objective = Objective(
            id=obj_id,
            description=description,
            objective_type=objective_type,
            target_value=target_value
        )
        self.state.objectives.append(objective)
        self.state.objectives_updated = True
        logger.info(f"Added objective: {description}")
        return obj_id
    
    def complete_objective(self, obj_id: str, progress_notes: str = ""):
        """Mark an objective as completed (storyline objectives cannot be manually completed)"""
        for obj in self.state.objectives:
            if obj.id == obj_id and not obj.completed:
                # Prevent manual completion of storyline objectives
                if obj.storyline:
                    logger.warning(f"Cannot manually complete storyline objective: {obj.description}. These are verified by emulator milestones.")
                    return False
                
                obj.completed = True
                obj.completed_at = datetime.now()
                obj.progress_notes = progress_notes
                self.state.objectives_updated = True
                logger.info(f"Completed objective: {obj.description}")
                return True
        return False
    
    def get_active_objectives(self) -> List[Objective]:
        """Get list of uncompleted objectives"""
        return [obj for obj in self.state.objectives if not obj.completed]
    
    def get_completed_objectives(self) -> List[Objective]:
        """Get list of completed objectives"""
        return [obj for obj in self.state.objectives if obj.completed]
    
    def check_objective_completion(self, game_state: Dict[str, Any]) -> List[str]:
        """Check if any objectives should be marked as completed based on game state"""
        completed_ids = []
        coords = self.get_player_coords(game_state)
        context = self.get_game_context(game_state)
        map_id = self.get_map_id(game_state)
        
        for obj in self.get_active_objectives():
            should_complete = False
            notes = ""
            
            if obj.objective_type == "location" and coords and obj.target_value:
                # Check if player reached target location
                # Note: target_value is a string (location name) for storyline objectives
                # Location objectives are completed via milestone verification, not coordinate checking
                # This section is for dynamically added coordinate-based objectives
                if isinstance(obj.target_value, (tuple, list)) and len(obj.target_value) == 2:
                    target_x, target_y = obj.target_value
                    if abs(coords[0] - target_x) <= 2 and abs(coords[1] - target_y) <= 2:
                        should_complete = True
                        notes = f"Reached location ({coords[0]}, {coords[1]})"
            
            elif obj.objective_type == "battle" and context == "battle":
                # Objective completed when battle starts
                should_complete = True
                notes = "Entered battle"
            
            elif obj.objective_type == "dialogue" and context == "dialogue":
                # Objective completed when dialogue starts
                should_complete = True
                notes = "Started dialogue"
            
            elif obj.objective_type == "map" and map_id and obj.target_value:
                # Check if player reached target map
                if map_id == obj.target_value:
                    should_complete = True
                    notes = f"Reached map {map_id}"
            
            if should_complete:
                self.complete_objective(obj.id, notes)
                completed_ids.append(obj.id)
        
        return completed_ids
    
    def check_storyline_milestones(self, game_state: Dict[str, Any]) -> List[str]:
        """Check emulator milestones and auto-complete corresponding storyline objectives"""
        completed_ids = []

        # Get milestones from the game state (if available)
        milestones = game_state.get("milestones", {})
        if not milestones:
            # No milestone data available, skip checking
            return completed_ids

        for obj in self.get_active_objectives():
            # Only check storyline objectives with milestone IDs
            if obj.storyline and obj.milestone_id and not obj.completed:
                # Check if the corresponding emulator milestone is completed
                milestone_completed = milestones.get(obj.milestone_id, {}).get("completed", False)

                if milestone_completed:
                    # Auto-complete the storyline objective
                    obj.completed = True
                    obj.completed_at = datetime.now()
                    obj.progress_notes = f"Auto-completed by emulator milestone: {obj.milestone_id}"
                    self.state.objectives_updated = True
                    self.story_objective_completed = True
                    completed_ids.append(obj.id)
                    logger.info(f"Auto-completed storyline objective via milestone {obj.milestone_id}: {obj.description}")

        return completed_ids
    
    def detect_stuck_pattern(self, coords: Optional[Tuple[int, int]], context: str, game_state: Dict[str, Any] = None) -> bool:
        """Detect if the agent appears to be stuck in a location/context"""
        # Don't trigger stuck detection during contexts where staying in place is expected
        if context in ["battle", "dialogue", "menu", "title"]:
            logger.debug(f"Skipping stuck detection - context: {context}")
            return False
        
        # Need valid coordinates for stuck detection
        if not coords or coords[0] is None or coords[1] is None:
            return False
        
        # Check for title sequence if game state is available
        if game_state:
            # Check if in title sequence (no player name or invalid coordinates)
            player_name = game_state.get("player", {}).get("name", "").strip()
            if not player_name or player_name == "????????":
                return False
                
            # Check if game state indicates title/intro
            game_state_value = game_state.get("game", {}).get("game_state", "").lower()
            if "title" in game_state_value or "intro" in game_state_value:
                return False
            
            # Check location for title sequence
            player_location = game_state.get("player", {}).get("location", "")
            if player_location == "TITLE_SEQUENCE":
                return False
            
        key = f"{coords[0]}_{coords[1]}_{context}"
        self.state.stuck_detection[key] = self.state.stuck_detection.get(key, 0) + 1
        
        # Consider stuck if we've been in the same location/context for 8+ consecutive steps
        return self.state.stuck_detection[key] >= 8
    
    def is_black_frame(self, frame) -> bool:
        """
        Check if the frame is mostly black (transition/loading screen).
        
        Args:
            frame: PIL Image or numpy array
            
        Returns:
            bool: True if frame is mostly black, False otherwise
        """
        try:
            
            # Convert to PIL Image if needed
            if hasattr(frame, 'convert'):  # It's already a PIL Image
                img = frame
            elif hasattr(frame, 'shape'):  # It's a numpy array
                img = Image.fromarray(frame)
            else:
                return False  # Unknown type, assume not black
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # Calculate the mean brightness
            # For RGB images, average across all channels
            if len(img_array.shape) == 3:
                mean_brightness = np.mean(img_array)
            else:
                mean_brightness = np.mean(img_array)
            
            # Also check the standard deviation to catch completely uniform frames
            std_dev = np.std(img_array)
            
            # A frame is considered "black" if:
            # 1. Mean brightness is very low (< 10 out of 255)
            # 2. OR standard deviation is very low (< 5) indicating uniform color
            is_black = mean_brightness < 10 or (mean_brightness < 30 and std_dev < 5)
            
            if is_black:
                logger.debug(f"Black frame detected: mean_brightness={mean_brightness:.2f}, std_dev={std_dev:.2f}")
            
            return is_black
            
        except Exception as e:
            logger.warning(f"Error checking for black frame: {e}")
            return False  # On error, assume not black to continue processing

    def _log_frame(self, frame):
        """Persist the current frame to disk for debugging."""
        if not frame or not self.frame_log_dir:
            return
        try:
            if hasattr(frame, 'save'):
                img = frame
            elif hasattr(frame, 'shape'):
                img = Image.fromarray(frame)
            else:
                # Unsupported type
                return

            os.makedirs(self.frame_log_dir, exist_ok=True)
            timestamp_ms = int(time.time() * 1000)
            step_id = self.state.step_counter
            filename = os.path.join(self.frame_log_dir, f"frame_{step_id:06d}_{timestamp_ms}.png")
            img.save(filename)
        except Exception as e:
            logger.debug(f"Failed to log frame: {e}")
    
    def get_relevant_history_summary(self, current_context: str, coords: Optional[Tuple[int, int]]) -> str:
        """Get a concise summary of relevant recent history"""
        # current_context and coords could be used for more sophisticated filtering in the future
        _ = current_context, coords  # Acknowledge unused parameters for now
        if not self.state.history:
            return "No previous history."
        
        # Get last N entries based on display count
        recent_entries = list(self.state.history)[-self.history_display_count:]
        
        # Format for LLM consumption
        summary_lines = []
        for i, entry in enumerate(recent_entries, 1):
            coord_str = f"({entry.player_coords[0]},{entry.player_coords[1]})" if entry.player_coords else "(?)"
            summary_lines.append(f"{i}. {entry.context} at {coord_str}: {entry.action_taken}")
        
        return "\n".join(summary_lines)
    
    def get_stuck_warning(self, coords: Optional[Tuple[int, int]], context: str, game_state: Dict[str, Any] = None) -> str:
        """Generate warning text if stuck pattern detected"""
        # Never show stuck warning in title sequence
        if context == "title":
            return ""
            
        if self.detect_stuck_pattern(coords, context, game_state):
            return "\n⚠️ WARNING: You appear to be stuck at this location/context. Try a different approach!\n" \
                   "💡 TIP: If you try an action like RIGHT but coordinates don't change from (X,Y) to (X+1,Y), there's likely an obstacle. Check the map around player P for walls (#) or other barriers blocking your path."
        return ""
    
    def create_game_state_summary(self, game_state: Dict[str, Any]) -> str:
        """Create a concise summary of the current game state"""
        try:
            game_info = game_state.get("game", {})
            
            summary_parts = []
            
            # Player location
            coords = self.get_player_coords(game_state)
            if coords:
                summary_parts.append(f"Player at ({coords[0]}, {coords[1]})")
            
            # Map info
            map_id = self.get_map_id(game_state)
            if map_id:
                summary_parts.append(f"Map {map_id}")
            
            # Context-specific info
            context = self.get_game_context(game_state)
            if context == "battle":
                summary_parts.append("In battle")
            elif context == "dialogue":
                dialogue_text = game_info.get("dialogue", {}).get("text", "")
                if dialogue_text:
                    summary_parts.append(f"Dialogue: {dialogue_text}")
            
            return " | ".join(summary_parts) if summary_parts else "Unknown state"
            
        except Exception as e:
            logger.warning(f"Error creating game state summary: {e}")
            return "Error reading state"
    
    def step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compatibility method for client that expects agent.step(game_state)
        
        Args:
            game_state: Complete game state dictionary (should include 'frame')
            
        Returns:
            Dictionary with 'action' and optional 'reasoning'
        """
        frame = game_state.get('frame')
        if frame is None:
            logger.error("🚫 No frame in game_state for SimpleAgent.step")
            return {"action": "WAIT", "reasoning": "No frame available"}
        
        action = self.process_step(frame, game_state)
        return {"action": action, "reasoning": "Simple agent decision"}

    def a_star(self, data, dest_x, dest_y):
        # Build tile map for quick lookup
        tile_map = {(tile['x'], tile['y']): tile for tile in data['tiles']}

        start_x = data['player_position']['x']
        start_y = data['player_position']['y']

        # Check if destination is walkable
        if (dest_x, dest_y) not in tile_map or not tile_map[(dest_x, dest_y)]['walkable']:
            return None  # Can't reach non-walkable destination

        # Heuristic: Manhattan distance
        def heuristic(x, y):
            return abs(x - dest_x) + abs(y - dest_y)

        # Priority queue: (f_score, counter, x, y, path)
        counter = 0
        pq = [(heuristic(start_x, start_y), counter, start_x, start_y, [])]
        visited = set()

        directions = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }

        while pq:
            f_score, _, x, y, path = heapq.heappop(pq)

            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Reached destination
            if x == dest_x and y == dest_y:
                return path

            # Explore neighbors
            for direction, (dx, dy) in directions.items():
                nx, ny = x + dx, y + dy

                if (nx, ny) in tile_map and tile_map[(nx, ny)]['walkable'] and (nx, ny) not in visited:
                    g_score = len(path) + 1
                    h_score = heuristic(nx, ny)
                    f = g_score + h_score

                    counter += 1
                    heapq.heappush(pq, (f, counter, nx, ny, path + [direction]))

        return ['A']  # No path found

    def build_interact_actions(self, json_data, dest_x, dest_y):
        coord = (dest_x, dest_y)
        tile_lookup = {(tile["x"], tile["y"]): tile for tile in json_data["tiles"]}
        tile = tile_lookup.get(coord)
        if tile:
            tile["walkable"] = True
            tile["type"] = tile.get("type") or "walkable"
        else:
            json_data["tiles"].append({
                "x": coord[0],
                "y": coord[1],
                "type": "walkable",
                "walkable": True
            })
        path = self.a_star(json_data, dest_x, dest_y)
        if not path:
            return []
        path.append("A")
        return path

    def reachable_tiles(self, json_data):
        """Return structured info about reachable tiles, highlighting doors/stairs/warps."""
        summary = {
            "total": 0,
            "tiles": [],
            "warps": [],
            "doors": [],
            "stairs": [],
            "text": "No reachable tiles"
        }

        if not json_data or not json_data.get("tiles") or not json_data.get("player_position"):
            return summary

        tile_map = {(tile["x"], tile["y"]): tile for tile in json_data["tiles"]}
        start = json_data["player_position"]
        if start is None or "x" not in start or "y" not in start:
            return summary

        start_coord = (start["x"], start["y"])
        if not tile_map.get(start_coord, {}).get("walkable", True):
            return summary

        visited = set([start_coord])
        queue = [start_coord]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            x, y = queue.pop(0)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                tile = tile_map.get((nx, ny))
                if tile and tile.get("walkable") and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        reachable_list = []
        warps = []
        doors = []
        stairs = []

        for coord in sorted(visited):
            tile = tile_map.get(coord, {})
            tile_type = tile.get("type", "walkable")
            entry = {
                "x": coord[0],
                "y": coord[1],
                "type": tile_type
            }
            if tile.get("warp_destination"):
                entry["warp_destination"] = tile["warp_destination"]
            reachable_list.append(entry)

            tile_type_lower = tile_type.lower()
            if tile.get("warp_destination") or "warp" in tile_type_lower:
                warps.append(entry)
            elif tile_type_lower == "door":
                doors.append(entry)
            elif tile_type_lower == "stairs":
                stairs.append(entry)

        summary["total"] = len(reachable_list)
        summary["tiles"] = reachable_list
        summary["warps"] = warps
        summary["doors"] = doors
        summary["stairs"] = stairs

        def _fmt_entry(entry):
            base = f"({entry['x']},{entry['y']})"
            if entry.get("warp_destination"):
                dest = entry["warp_destination"]
                dest_name = dest.get("to_name", "Unknown")
                target = dest.get("to_pos")
                if target:
                    base += f" → {dest_name} ({target[0]},{target[1]})"
                else:
                    base += f" → {dest_name}"
            return base

        lines = []

        conductive_tiles = []
        for entry in reachable_list:
            tile_type = entry.get("type", "").lower()
            if tile_type not in {"door", "stairs"} and not entry.get("warp_destination"):
                conductive_tiles.append(_fmt_entry(entry))

        if conductive_tiles:
            for coord in conductive_tiles:
                lines.append(f"{coord}")

        if doors:
            lines.append("Doors:")
            for entry in doors:
                lines.append(f"{_fmt_entry(entry)}")
        if stairs:
            lines.append("Stairs:")
            for entry in stairs:
                lines.append(f"{_fmt_entry(entry)}")
        if warps:
            lines.append("Warps:")
            for entry in warps:
                lines.append(f"{_fmt_entry(entry)}")

        summary["text"] = "\n".join(lines)

        return summary

    def process_step(self, frame, game_state: Dict[str, Any]) -> str:
        """
        Main processing step for simple mode with history tracking
        
        Args:
            frame: Current game frame (PIL Image or similar)
            game_state: Complete game state dictionary
            
        Returns:
            Action string or list of actions
        """
        # CRITICAL: Validate frame before any VLM processing
        if frame is None:
            logger.error("🚫 CRITICAL: SimpleAgent.process_step called with None frame - cannot proceed")
            return "WAIT"
        
        # Validate frame is a proper image
        if not (hasattr(frame, 'save') or hasattr(frame, 'shape')):
            logger.error(f"🚫 CRITICAL: SimpleAgent.process_step called with invalid frame type {type(frame)} - cannot proceed")
            return "WAIT"
        
        # Additional PIL Image validation
        if hasattr(frame, 'size'):
            width, height = frame.size
            if width <= 0 or height <= 0:
                logger.error(f"🚫 CRITICAL: SimpleAgent.process_step called with invalid frame size {width}x{height} - cannot proceed")
                return "WAIT"
        
        # Check for black frame (transition screen)
        if self.is_black_frame(frame):
            logger.info("⏳ Black frame detected (likely a transition), waiting for next frame...")
            return "WAIT"  # Return WAIT to skip this frame and wait for the next one

        # Save the current frame for debugging
        self._log_frame(frame)
        
        try:
            # Increment step counter
            self.state.step_counter += 1
            
            # Log full game state for debugging
            try:
                from utils.llm_logger import get_llm_logger
                llm_logger = get_llm_logger()
                if llm_logger:
                    llm_logger.log_state_snapshot(game_state, self.state.step_counter)
            except Exception as e:
                logger.debug(f"Failed to log state snapshot: {e}")
            
            # Get current state info
            coords = self.get_player_coords(game_state)
            context = self.get_game_context(game_state)
            map_id = self.get_map_id(game_state)
            
            # Format the current state for LLM (includes movement preview)
            formatted_state = format_state_for_llm(game_state)
            
            # Get movement memory for the current area
            movement_memory = ""
            if coords:
                movement_memory = self.get_area_movement_memory(coords)
            
            # Check for objective completion first
            self.check_objective_completion(game_state)
            
            # Check storyline milestones and auto-complete objectives
            self.check_storyline_milestones(game_state)

            
            # Get relevant history and stuck detection
            history_summary = self.get_relevant_history_summary(context, coords)
            stuck_warning = self.get_stuck_warning(coords, context, game_state)
            recent_actions_str = ', '.join(list(self.state.recent_actions)[-5:]) if self.state.recent_actions else 'None'
            
            # Format objectives for LLM
            active_objectives = self.get_active_objectives()
            completed_objectives_list = self.get_completed_objectives()
            objectives_summary = self._format_objectives_for_llm(active_objectives, completed_objectives_list)
            added_objectives_summary, any_active = self._format_added_objectives_for_llm(active_objectives, completed_objectives_list)
            active_added_objectives_summary, any_active = self._format_active_added_objectives_for_llm(active_objectives, completed_objectives_list)
            # Build pathfinding rules section (only if not in title sequence)
            map_preview = format_movement_preview_for_llm(game_state)
            player_coords = coords or self.get_player_coords(game_state)
            if player_coords:
                coord_str = f"(X={player_coords[0]}, Y={player_coords[1]})"
                map_preview = map_preview.replace(
                    "This is also a movement preview showing you a summary of your immediately available movements:\n",
                    ""
                )
            current_player_coords = f"(X={player_coords[0]}, Y={player_coords[1]})"
            map_info = game_state.get('map', {}) or {}
            player_data = game_state.get('player', {}) or {}
            map_only_sections, json_data = _format_map_info(map_info, player_data, include_npcs=True, full_state_data=game_state, use_json_map=True)
            map_only = "\n".join(map_only_sections) if map_only_sections else ""
            recent_coords = [entry.player_coords for entry in list(self.state.history)[-5:]]

            def calculate_blocked_tiles(prev_coord, curr_coord, last_actions):
                if 'A' in last_actions:
                    return []

                directions = {
                    'UP': (0, -1),
                    'DOWN': (0, 1),
                    'LEFT': (-1, 0),
                    'RIGHT': (1, 0)
                }

                curr_x, curr_y = curr_coord  # where we actually ended up
                expected_x, expected_y = prev_coord  # where we started

                for action in last_actions:
                    if action not in directions:
                        return []
                    dx, dy = directions[action]
                    next_expected_x = expected_x + dx
                    next_expected_y = expected_y + dy

                    if expected_x == curr_x and expected_y == curr_y:
                        # We're at the current position, so the NEXT tile must be blocked
                        return [(next_expected_x, next_expected_y)]

                    # Move to next expected position
                    expected_x, expected_y = next_expected_x, next_expected_y

                # If we got through all actions and never matched curr_coord,
                # either we made it to the end or something unexpected happened
                return []
            blocked_tiles = []
            if len(recent_coords) > 0 and len(self.last_turn_actions) > 0 and player_coords:
                blocked_tiles = calculate_blocked_tiles(recent_coords[-1], player_coords, self.last_turn_actions)

            if blocked_tiles and json_data and json_data.get("tiles"):
                blocked_set = set(blocked_tiles)
                tile_lookup = {(tile["x"], tile["y"]): tile for tile in json_data["tiles"]}
                for coord in blocked_set:
                    tile = tile_lookup.get(coord)
                    if tile:
                        tile["walkable"] = False
                        tile["type"] = tile.get("type") or "blocked"
                    else:
                        json_data["tiles"].append({
                            "x": coord[0],
                            "y": coord[1],
                            "type": "blocked",
                            "walkable": False
                        })

            reachable_tiles_info = self.reachable_tiles(json_data)
            reachable_tiles_text = reachable_tiles_info["text"]

            player_location = game_state.get("player", {}).get("location", "Unknown Location")
            if player_location == 'Map_18_0B':
                player_location = 'PETALBURG_WOODS'
            pathfinding_rules = ""
            if context != "title":
                pathfinding_rules = ""
            loop_warning = ""
            if len(set(recent_coords[-6:])) <= 3 and len(recent_coords) >= 6:
                loop_warning = "⚠️ You are revisiting the same coordinates repeatedly. Pick a direction you haven't tried yet (use the map and movement preview)."
            if len(self.prev_analysis) > 0:
                prev_analysis = self.prev_analysis[-1]
            else:
                prev_analysis = "No previous analysis yet"

            recent_responses = list(self.response_history)[-3:]  # or whatever count you want
            prev_responses_str = "\n".join(
                f"{resp.strip()}\n{'=' * 80}"
                for resp in recent_responses
            )
            prev_responses_str = prev_responses_str.rstrip("=\n")  # optional to drop trailing bar
            image_recognition_prompt = f"""
            
You are playing pokemon emerald. Does this image contain a dialogue box?

If it does, respond only with one word: YES

"""
            if len(self.interact_destination_list) > 0: #interaction list exists, start searching
                if frame and (hasattr(frame, 'save') or hasattr(frame, 'shape')):
                    print("🔍 Making VLM objectives call...")
                    try:
                        response = self.vlm.get_query(frame, image_recognition_prompt, "simple_mode", model_name = 'gemini-2.5-pro')
                        print(f"🔍 VLM response received: {response[:100]}..." if len(
                            response) > 100 else f"🔍 VLM response: {response}")
                    except Exception as e:
                        print(f"❌ VLM call failed: {e}")
                        return "WAIT"
                else:
                    logger.error("🚫 CRITICAL: About to call VLM but frame validation failed - this should never happen!")
                    return "WAIT"
                normalized = response.strip().upper() if response else ""
                if normalized == 'YES':
                    self.interact_destination_list = []
                else: #didn't recognize an image, try again.
                    dest_x, dest_y = self.interact_destination_list.pop()
                    path = self.build_interact_actions(dest_x, dest_y, json_data)
                    return path


            planning_prompt = f"""
You are the planning module for a pokemon emerald agent speedrun scaffolding. 

Your goal is to use your knowledge of pokemon emerald to add intermediary objectives with navigation tips that help the action agent accomplish its goals and finish the game. Make sure to only include objectives that directly help you accomplish the next goals!
Your goal is to complete the game as fast as possible, so make sure each objective is clear, direct, useful, and informative.

You will be called after every story objective to add objectives to assist the agent to get to the direct next story objective, you have the most crucial role in the entire scaffolding!

Think about common failure modes for pokemon agents. Sometimes they need explicitly directional hints and context to avoid loops or missing the right path!
Also try to break up big objectives into smaller parts, giving detailed steps and directions that the agent can complete along the way.
Provide essential steps and break up large objectives into smaller sequential parts! The model will be able to complete these objectives sequentially.
The action agent cannot directly see the story objectives, so make sure to include sub objectives for ALL parts of the main story objective. Each building you have to enter, each NPC you have to talk to, should each have its own objective.
Be thorough with adding all the required sub objectives for the next story objective.

You also have access to the current game frame. Visually inspect it to get a sense of your current location and context.

Current story objectives you are trying to accomplish:
{objectives_summary}

These are the sub-objectives you have added. Avoid duplicate objectives here:
{added_objectives_summary}

Current location:
{player_location}

When your next objective is to confront roxanne, make sure to add some objectives to catch additional pokemon!

You should format your response as follows.

OBJECTIVES:
[Review your current objectives. You have main storyline objectives (story_*) that track overall Emerald progression - these are automatically verified and you CANNOT manually complete them. 
You also have access to the following command in this section to sub-objectives: ADD_OBJECTIVE: type:description:target_value (e.g., "ADD_OBJECTIVE: location:Find Pokemon Center in town:(15,20)" or "ADD_OBJECTIVE: item:Buy Pokeballs:5"). The action model will be able to manually complete these objectives
This section should only contain calls do the ADD_OBJECTIVE tool at the start of each line, eg
ADD_OBJECTIVE: location:Find Pokemon Center in town:(15,20). You should only add objectives that directly help you accomplish the next story goal.
Also, avoid duplicate goals here. They will take up unnecessary precious space in the limited goal space.
Only include essential goals. Goals like "level up your pokemon" or "get a free potion" are not helpful.]
**IMPORTANT** be very specific with your directions. Eg when designating buildings, prefer directions like "south west, last building" over generic "west side of town".
"""

            self_critique_prompt = f"""
You are managing an action agent for pokemon emerald in a pokemon emerald speedrun. You are the self critique module. You should examine the current objectives, the analysis history of the planning agent, and use your knowledge of pokemon emerald to detect loops, mistaken assumptions, and provide guidance for the action module to complete the main story objectives.
You are called at the start of a turn for the LLM, so the actions you are seeing have already occured. 
These are the previous responses:
"""
            # Make VLM call for planning module - double-check frame validation before VLM
            self_critique_response = ""
            if self.state.step_counter == 0:
                return "WAIT"

            if self.state.step_counter == 1 or not any_active:
                #self._complete_all_added_objectives("Story milestone reached - refreshing planner objectives")
                if frame and (hasattr(frame, 'save') or hasattr(frame, 'shape')):
                    print("🔍 Making VLM objectives call...")
                    try:
                        response = self.vlm.get_query(frame, planning_prompt, "simple_mode", model_name = 'gemini-2.5-pro')
                        print(f"🔍 VLM response received: {response[:100]}..." if len(
                            response) > 100 else f"🔍 VLM response: {response}")
                    except Exception as e:
                        print(f"❌ VLM call failed: {e}")
                        return "WAIT"
                else:
                    logger.error("🚫 CRITICAL: About to call VLM but frame validation failed - this should never happen!")
                    return "WAIT"
                # will automatically update objectives
                actions, reasoning, analysis = self._parse_structured_response(response, game_state, json_data=json_data)
            self.story_objective_completed = False
            if self.state.step_counter < 2:
                return "WAIT"

            # Create enhanced prompt with objectives, history context and chain of thought request
            prompt = f"""You are playing as the Protagonist Brendan in Pokemon Emerald. 
            Based on the current game frame and state information, think through your next move and choose the best action.

Hint: Use the reachable tiles, map preview, and visual frame to determine which coordinate you want to go to, then use the navigate_to(x,y) action to find the optimal path to your destination.

ALSO IMPORTANT: Use the interact_with(x,y) tool to interact with objects and NPCs. You don't need to be near the object to use this tool, it will navigate towards it for you.

These are the sub-objectives added by the planning agent. These will help you accomplish the main story objectives:
{active_added_objectives_summary}

This is your analysis from your previous turn:
{prev_analysis}

Your current location is:
{player_location}

The current reachable tiles from your location are:
{reachable_tiles_text}

Movement preview (check this to make sure you aren't selecting a blocked action):
{map_preview}
IMPORTANT: The movement preview doesn't show NPCs, so look for visual confirmation if you think an NPC is blocking your path. If you are blocked by an NPC you should move around them, they only block a single tile. If you need to complete a story segment to move an npc, it will show up in your objectives.

Your most recent actions are:
{recent_actions_str}

And your current coordinates:
{current_player_coords}


Available actions: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, navigate_to(x,y), interact_with(x,y)
Remember: if you want to interact with an object or npc, you should always use the interact_with(x,y) tool!!
Do not select a movement that is blocked. REMEMBER, BROWN LEDGES ARE BLOCKED!

**IMPORTANT** To enter doors/stairs/warps CHECK THE MOVEMENT PREVIEW AND USE SINGLE ACTIONS. navigate_to is great for long distances, but it can struggle with entering locations if you are not perfectly aligned with the door or you are stuck on an NPC. The movement preview will have better information for you to use.
If you want to you can use navigate_to to get close to the door, but afterwards make sure to use single actions.

Navigation tips:
- Enter and exit locations through the middle rather than hugging a specific side, this maximizes visibility of the area to locate the exit.
- Unless you are blocked, use the navigate_to() tool as much as possible. This is much faster than single steps.
In your response include the following sections:

OBJECTIVES:
[Make sure to review your current objectives. These are your highest priority, everything you do should be in service of accomplishing these goals
You can also Complete these objectives: COMPLETE_OBJECTIVE: objective_id:notes (e.g., "COMPLETE_OBJECTIVE: my_sub_obj_123:Successfully bought Pokeballs")
This section should only contain calls to the tools at the start of each line, eg
COMPLETE_OBJECTIVE: my_sub_obj_123:Successfully bought Pokeballs
Important: You cannot add objectives, only complete them.
Make sure you are confident you successfully completed the objectives before marking them complete. For example if you have an objective to talk to an NPC, you should only
mark the objective complete on the turn you see the dialogue box with that NPC. You should also visually confirm you entered the right location before marking location objectives as complete.
Don't mark them pre-emptively as you are executing the action.]

ACTION:
[If you are in dialogue or battle, prefer single actions like 'A'.
You also have access to the navigate_to(x,y) tool. This will automatically run A* on your selected coordinate to find the optimal way to reach your destination. This is a powerful tool that should be used
liberally! Using it will also override any other actions you input, so don't include it with other actions.
When using the navigate_to(x,y) tool, you can only choose from the reachable tiles! Other tiles are NOT ALLOWED, YOU WILL BE STUCK. Always check the list of reachable tiles! The list of reachables tiles is provided above. Lower X means the tile is to your west, while a lower Y means the tile is to your north.
Some important notes: Don't use the navigate_to(x,y) tool when you are in dialogue or battle, you will be stuck.
IMPORTANT RULE: Don't interact with NPCs unless you have to. It will likely waste time.
ALSO IMPORTANT: You interact with warps/stairs by walking into them, not pressing 'A'. They will also show up in your movement preview. Confirm you are in front of them using your movement preview, then walk into them to transition.
To interact with NPCs/Objects you also have access to an interact_with(x,y) tool. You can choose a traversable coordinate and if there is an NPC or object there, it will take you there and you will interact with it. You do not have to be next to the object, the tool will navigate for you!
***IMPORTANT RULE***: DO NOT use the interact_with(x,y) tool on any coordinates you previously used it on. CHECK YOUR PREVIOUS ACTIONS FIRST! You likely chose the wrong coordinate if you repeat them!!
ANOTHER IMPORTANT RULE: When interacting with NPCs, they are present in the traversable tile list. Select from those!!!
Before choosing your action, inspect your frame. You have a tendency to get stuck on the "Got away safely!" image and stop recognizing you are stuck in dialogue.]

ANALYSIS:
[Summarize your current situation. This will be passed onto you as context during your next turn. It's especially important to summarize any dead ends you found and potential alternate paths. This is the only information that gets passed forward in time, so note anything important here. You can be as verbose as you like.
Very important: Avoid mentioning coordinates at all here, you tend to hallucinate and confuse yourself, ending up in an eternal loop]
Context: {context} """
            
            # Print complete prompt to terminal for debugging
            '''
            print("\n" + "="*120)
            print("🤖 SIMPLE AGENT PROMPT SENT TO VLM:")
            print("="*120)
            
            # Print prompt in chunks to avoid terminal truncation
            sys.stdout.write(prompt)
            sys.stdout.write("\n")
            sys.stdout.flush()
            
            print("="*120)
            print("🤖 END OF SIMPLE AGENT PROMPT")
            print("="*120 + "\n")
            sys.stdout.flush()'''
            
            # Make VLM call - double-check frame validation before VLM
            if frame and (hasattr(frame, 'save') or hasattr(frame, 'shape')):
                print("🔍 Making VLM call...")
                try:
                    response = self.vlm.get_query(frame, prompt, "simple_mode", model_name = "gemini-2.5-pro")
                    print(f"🔍 VLM response received: {response[:100]}..." if len(response) > 100 else f"🔍 VLM response: {response}")
                except Exception as e:
                    print(f"❌ VLM call failed: {e}")
                    return "WAIT"
            else:
                logger.error("🚫 CRITICAL: About to call VLM but frame validation failed - this should never happen!")
                return "WAIT"
            if response:
                self.response_history.append(response)
            # Extract action(s) from structured response
            actions, reasoning, analysis = self._parse_structured_response(response, game_state, json_data = json_data)

            self.last_turn_actions = actions

            self.prev_analysis.append(analysis)
            if len(self.prev_analysis) >= 2 and self.prev_analysis[-2] == self.prev_analysis[-1]:
                self.prev_analysis = ["I seem to be stuck in a loop. I should carefully examine my situation and choose a different action this turn"]
            # Check for failed movement by comparing previous coordinates
            if len(self.state.history) > 0:
                prev_coords = self.state.history[-1].player_coords
                if prev_coords and coords:
                    # If coordinates didn't change and we attempted a movement, record it as failed
                    if (prev_coords == coords and 
                        isinstance(actions, list) and len(actions) > 0 and 
                        actions[0] in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
                        self.record_failed_movement(coords, actions[0], "movement_blocked")
                    elif (prev_coords == coords and 
                          isinstance(actions, str) and 
                          actions in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
                        self.record_failed_movement(coords, actions, "movement_blocked")

            # Record this step in history with reasoning
            game_state_summary = self.create_game_state_summary(game_state)
            action_with_reasoning = f"{actions} | Reasoning: {reasoning}" if reasoning else str(actions)
            history_entry = HistoryEntry(
                timestamp=datetime.now(),
                player_coords=coords,
                map_id=map_id,
                context=context,
                action_taken=action_with_reasoning,
                game_state_summary=game_state_summary
            )
            self.state.history.append(history_entry)
            
            # Update recent actions
            if isinstance(actions, list):
                self.state.recent_actions.extend(actions)
                # Increment movement memory action counter by number of actions
                self.state.movement_memory_action_counter += len(actions)
            else:
                self.state.recent_actions.append(actions)
                # Increment movement memory action counter
                self.state.movement_memory_action_counter += 1
            
            # Check if we should clear movement memory
            if (self.movement_memory_clear_interval > 0 and 
                self.state.movement_memory_action_counter >= self.movement_memory_clear_interval):
                logger.info(f"🧹 Movement memory clear triggered after {self.state.movement_memory_action_counter} actions")
                # Use partial clear to keep some recent memory
                self.clear_movement_memory(partial=True)
            
            # Reset stuck detection for other locations when we move
            if coords:
                keys_to_reset = [k for k in self.state.stuck_detection.keys() 
                               if not k.startswith(f"{coords[0]}_{coords[1]}")]
                for key in keys_to_reset:
                    if self.state.stuck_detection[key] > 0:
                        self.state.stuck_detection[key] = max(0, self.state.stuck_detection[key] - 1)
            
            # Update server with agent step and metrics (for agent thinking display)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error in simple agent processing: {e}")
            return ["A"]  # Default safe action as list
    
    def _parse_actions(self, response: str, game_state: Dict[str, Any] = None, json_data = None) -> List[str]:
        """Parse action response from LLM into list of valid actions"""

        nav_match = re.search(r"navigate_to\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
                              response, flags=re.IGNORECASE)


        if nav_match and json_data:
            dest_x, dest_y = map(int, nav_match.groups())

            path = self.a_star(json_data, dest_x, dest_y)
            if path:
                return path

        interact_match = re.search(r"interact_with\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
                                   response, flags=re.IGNORECASE)
        if interact_match and json_data:
            directions = {
                'UP': (0, -1),
                'DOWN': (0, 1),
                'LEFT': (-1, 0),
                'RIGHT': (1, 0),
                'UP_LEFT': (-1, -1),
                'UP_RIGHT': (1, -1),
                'DOWN_LEFT': (-1, 1),
                'DOWN_RIGHT': (1, 1)
            }

            dest_x, dest_y = map(int, interact_match.groups())
            self.interact_destination_list.clear()
            for dx, dy in directions.values():
                self.interact_destination_list.append((dest_x + dx, dest_y + dy))
            path = self.build_interact_actions(dest_x, dest_y, json_data)
            if path:
                return path
            else:
                logger.warning(f"interact_with({dest_x},{dest_y}) failed to find a path; falling back to raw actions")


        response_upper = response.upper().strip()
        valid_actions = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
        
        # Parse multiple actions (could be comma or space separated)
        actions_found = []
        # Replace commas with spaces for consistent parsing
        response_clean = response_upper.replace(',', ' ').replace('.', ' ')
        tokens = response_clean.split()

        for token in tokens:
            if token in valid_actions:
                actions_found.append(token)
                if len(actions_found) >= 10:  # Max 10 actions
                    break
        
        # Validate movement sequences if we have game state
        '''
        if game_state and len(actions_found) > 1:
            # Check if this is a movement sequence
            movement_actions = [a for a in actions_found if a in ['UP', 'DOWN', 'LEFT', 'RIGHT']]
            if movement_actions:
                # Validate the movement sequence
                is_valid, reason = self.validate_movement_sequence(movement_actions, game_state)
                if not is_valid:
                    logger.warning(f"Movement sequence validation failed: {reason}")
                    # Only take the first movement if sequence is invalid
                    if movement_actions:
                        actions_found = [movement_actions[0]]
                        logger.info(f"Reduced to single movement: {actions_found[0]}")'''
        
        # If no valid actions found, use default
        if not actions_found:
            actions_found = ['A']
        
        return actions_found
    
    def _format_objectives_for_llm(self, active_objectives: List[Objective], completed_objectives: List[Objective]) -> str:
        """Format objectives for LLM consumption"""
        lines = []
        
        storyline_active = [obj for obj in active_objectives if obj.storyline]
        if storyline_active:
            lines.append("🎯 ACTIVE STORY OBJECTIVES:")
            for idx, obj in enumerate(storyline_active[:2], 1):
                target_str = f" (Target: {obj.target_value})" if obj.target_value else ""
                lines.append(f"  {idx}. [{obj.objective_type}] {obj.description}{target_str} [ID: {obj.id}]")
        else:
            lines.append("YOUR STORY OBJECTIVES: None - Consider setting some goals!")
        
        storyline_completed = [obj for obj in completed_objectives if obj.storyline]
        if storyline_completed:
            lines.append("✅ RECENTLY COMPLETED STORY OBJECTIVES:")
            for obj in storyline_completed[-3:]:
                lines.append(f"  ✓ [{obj.objective_type}] {obj.description}")
        
        return "\n".join(lines)

    def _format_added_objectives_for_llm(self, active_objectives: List[Objective],
                                   completed_objectives: List[Objective]) -> str:
        """Format objectives for LLM consumption"""
        lines = []
        added_active = [obj for obj in active_objectives if not obj.storyline]
        any_active = True
        if added_active:
            lines.append("🎯 ACTIVE SUB OBJECTIVES:")
            for idx, obj in enumerate(added_active[:5], 1):
                target_str = f" (Target: {obj.target_value})" if obj.target_value else ""
                lines.append(f"  {idx}. [{obj.objective_type}] {obj.description}{target_str} [ID: {obj.id}]")
        else:
            lines.append("🎯 ACTIVE SUB OBJECTIVES: None - Consider adding some objectives!")
            any_active = False

        added_completed = [obj for obj in completed_objectives if not obj.storyline]
        if added_completed:
            lines.append("✅ RECENTLY COMPLETED SUB OBJECTIVES:")
            for obj in added_completed[-3:]:
                lines.append(f"  ✓ [{obj.objective_type}] {obj.description}")

        return "\n".join(lines), any_active


    def _format_active_added_objectives_for_llm(self, active_objectives: List[Objective],
                                   completed_objectives: List[Objective]) -> str:
        """Format objectives for LLM consumption"""
        lines = []
        added_active = [obj for obj in active_objectives if not obj.storyline]
        any_active = True
        if added_active:
            lines.append("🎯 ACTIVE SUB OBJECTIVES:")
            for idx, obj in enumerate(added_active[:5], 1):
                target_str = f" (Target: {obj.target_value})" if obj.target_value else ""
                lines.append(f"  {idx}. [{obj.objective_type}] {obj.description}{target_str} [ID: {obj.id}]")
        else:
            lines.append("🎯 ACTIVE SUB OBJECTIVES: None - Consider adding some objectives!")
            any_active = False

        return "\n".join(lines), any_active
    
    def _parse_structured_response(self, response: str, game_state: Dict[str, Any] = None, json_data = None) -> Tuple[List[str], str]:
        """Parse structured chain-of-thought response and extract actions and reasoning"""
        try:
            # Extract sections from structured response
            analysis = ""
            objectives_section = ""
            plan = ""
            reasoning = ""
            actions = []
            
            # Split response into lines for processing
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # Identify section headers
                if line.upper().startswith('ANALYSIS:'):
                    current_section = 'analysis'
                    analysis = line[9:].strip()  # Remove "ANALYSIS:" prefix
                elif line.upper().startswith('OBJECTIVES:'):
                    current_section = 'objectives'
                    objectives_section = line[11:].strip()  # Remove "OBJECTIVES:" prefix
                elif line.upper().startswith('PLAN:'):
                    current_section = 'plan'
                    plan = line[5:].strip()  # Remove "PLAN:" prefix
                elif line.upper().startswith('REASONING:'):
                    current_section = 'reasoning'
                    reasoning = line[10:].strip()  # Remove "REASONING:" prefix
                elif line.upper().startswith('MEMORIES:'):
                    current_section = 'memories'
                    memory = line[9:].strip()
                    if len(memory) > 5: #
                        self.memories.append(memory)
                elif line.upper().startswith('ACTION:'):
                    current_section = 'action'
                    # Extract actions from this line
                    action_text = line[7:].strip()  # Remove "ACTION:" prefix
                    if action_text:  # Only parse if there's content
                        actions = self._parse_actions(action_text, game_state, json_data = json_data)
                elif line and current_section:
                    # Continue content of current section
                    if current_section == 'analysis':
                        analysis += " " + line
                    elif current_section == 'objectives':
                        objectives_section += "\n" + line
                    elif current_section == 'plan':
                        plan += " " + line
                    elif current_section == 'reasoning':
                        reasoning += " " + line
                    elif current_section == 'memories':
                        self.memories.append(line)
                    elif current_section == 'action':
                        # Additional action parsing from action section content
                        if line.strip():  # Only process non-empty lines
                            additional_actions = self._parse_actions(line, game_state, json_data = json_data)
                            actions.extend(additional_actions)
                            if len(actions) >= 10:  # Max 10 actions
                                actions = actions[:10]
                                break
            
            # Process objectives if mentioned
            if objectives_section:
                self._process_objectives_from_response(objectives_section)
            
            # If no actions found in structured format, fall back to parsing entire response
            if not actions:
                actions = self._parse_actions(response, game_state, json_data = json_data)
            
            # Create concise reasoning summary
            reasoning_parts = []
            if analysis:
                reasoning_parts.append(f"Analysis: {analysis}")
            if objectives_section:
                reasoning_parts.append(f"Objectives: {objectives_section}")
            if plan:
                reasoning_parts.append(f"Plan: {plan}")
            if reasoning:
                reasoning_parts.append(f"Reasoning: {reasoning}")
            
            full_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No reasoning provided"
            
            return actions, full_reasoning, analysis
            
        except Exception as e:
            logger.warning(f"Error parsing structured response: {e}")
            # Fall back to basic action parsing
            return self._parse_actions(response, game_state, json_data), "Error parsing reasoning", ""
    
    def _process_objectives_from_response(self, objectives_text: str):
        """Process objective management commands from LLM response"""
        try:
            # Look for ADD_OBJECTIVE and COMPLETE_OBJECTIVE commands
            for line in objectives_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                upper_line = line.upper()

                if upper_line.startswith('ADD_OBJECTIVE:'):
                    # Parse format: ADD_OBJECTIVE: type:description:target_value
                    content = line[14:].strip()  # Remove prefix
                    parts = content.split(':', 2)  # Split into max 3 parts
                    
                    if len(parts) >= 2:
                        obj_type = parts[0].strip()
                        description = parts[1].strip()
                        target_value = parts[2].strip() if len(parts) > 2 else None
                        
                        # Parse target_value based on type
                        #parsed_target = self._parse_target_value(obj_type, target_value)
                        
                        # Add the objective
                        self.add_objective(description, obj_type, target_value)
                
                elif upper_line.startswith('COMPLETE_OBJECTIVE:'):
                    # Parse format: COMPLETE_OBJECTIVE: objective_id:notes
                    content = line[19:].strip()  # Remove prefix
                    parts = content.split(':', 1)  # Split into max 2 parts
                    
                    if len(parts) >= 1:
                        obj_id = parts[0].strip()
                        notes = parts[1].strip() if len(parts) > 1 else "Manually completed by LLM"
                        
                        # Complete the objective
                        success = self.complete_objective(obj_id, notes)
                        if success:
                            logger.info(f"LLM manually completed objective: {obj_id}")
                        else:
                            logger.warning(f"LLM tried to complete non-existent or already completed objective: {obj_id}")
                        
        except Exception as e:
            logger.warning(f"Error processing objectives from response: {e}")
    
    def _parse_target_value(self, obj_type: str, target_str: Optional[str]) -> Any:
        """Parse target value based on objective type"""
        if not target_str:
            return None
            
        try:
            if obj_type == "location":
                # Try to parse coordinates like "(15,20)" or "15,20"
                target_str = target_str.strip('()')
                if ',' in target_str:
                    x, y = map(int, target_str.split(','))
                    return (x, y)
            elif obj_type == "map":
                # Try to parse map ID as integer
                return int(target_str)
            else:
                # For other types, return as string
                return target_str
        except (ValueError, TypeError):
            # If parsing fails, return as string
            return target_str
    
    def get_memory_usage_estimate(self) -> Dict[str, int]:
        """Estimate current memory usage for context management"""
        history_chars = sum(len(str(entry)) for entry in self.state.history)
        recent_actions_chars = sum(len(action) for action in self.state.recent_actions)
        objectives_chars = sum(len(f"{obj.description} {obj.target_value}") for obj in self.state.objectives)
        
        return {
            "history_entries": len(self.state.history),
            "history_chars": history_chars, 
            "recent_actions": len(self.state.recent_actions),
            "recent_actions_chars": recent_actions_chars,
            "objectives_count": len(self.state.objectives),
            "objectives_chars": objectives_chars,
            "estimated_total_chars": history_chars + recent_actions_chars + objectives_chars
        }
    
    def get_objectives_state(self) -> Dict[str, Any]:
        """Get objectives formatted for forwarding in game state"""
        return {
            "active": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "type": obj.objective_type,
                    "target": obj.target_value,
                    "created_at": obj.created_at.isoformat()
                }
                for obj in self.get_active_objectives()
            ],
            "completed": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "type": obj.objective_type,
                    "target": obj.target_value,
                    "completed_at": obj.completed_at.isoformat() if obj.completed_at else None,
                    "notes": obj.progress_notes
                }
                for obj in self.get_completed_objectives()[-5:]  # Last 5 completed
            ],
            "updated": self.state.objectives_updated
        }
    
    def trim_history_for_context(self, max_chars: int = 4000):
        """Trim history to fit within context limits"""
        # Preserve minimum history for context
        min_history = max(5, self.history_display_count // 2)
        min_actions = max(10, self.actions_display_count // 2)
        
        while self.get_memory_usage_estimate()["estimated_total_chars"] > max_chars and len(self.state.history) > min_history:
            self.state.history.popleft()
            
        while len(self.state.recent_actions) > min_actions and self.get_memory_usage_estimate()["estimated_total_chars"] > max_chars:
            self.state.recent_actions.popleft()
    
    def reset_objectives_updated_flag(self):
        """Reset the objectives updated flag (call after forwarding state)"""
        self.state.objectives_updated = False
    
    def configure_history_limits(self, max_history_entries: int = None, max_recent_actions: int = None, 
                                history_display_count: int = None, actions_display_count: int = None,
                                movement_memory_clear_interval: int = None):
        """Configure history tracking parameters at runtime"""
        if max_history_entries is not None:
            # Create new deque with updated max length, preserving existing data
            existing_history = list(self.state.history)
            self.state.history = deque(existing_history, maxlen=max_history_entries)
            
        if max_recent_actions is not None:
            # Create new deque with updated max length, preserving existing data
            existing_actions = list(self.state.recent_actions)
            self.state.recent_actions = deque(existing_actions, maxlen=max_recent_actions)
            
        if history_display_count is not None:
            self.history_display_count = history_display_count
            
        if actions_display_count is not None:
            self.actions_display_count = actions_display_count
        
        if movement_memory_clear_interval is not None:
            self.movement_memory_clear_interval = movement_memory_clear_interval
        
        logger.info(f"Updated history configuration: {len(self.state.history)}/{self.state.history.maxlen} history, "
                   f"{len(self.state.recent_actions)}/{self.state.recent_actions.maxlen} actions, "
                   f"display {self.history_display_count}/{self.actions_display_count}, "
                   f"movement memory clear interval: {self.movement_memory_clear_interval}")
    
    def load_history_from_llm_checkpoint(self, checkpoint_file: str):
        """Load SimpleAgent history from LLM checkpoint file"""
        try:
            from utils.llm_logger import get_llm_logger
            import json
            import re
            from datetime import datetime
            
            if not os.path.exists(checkpoint_file):
                logger.info(f"No checkpoint file found: {checkpoint_file}")
                return False
            
            # Use LLM logger to restore cumulative metrics first
            llm_logger = get_llm_logger()
            if llm_logger:
                restored_step_count = llm_logger.load_checkpoint(checkpoint_file)
                if restored_step_count is not None:
                    logger.info(f"✅ LLM logger restored checkpoint with {restored_step_count} steps")
                    # Update SimpleAgent step counter to match LLM logger
                    self.state.step_counter = restored_step_count
            
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            log_entries = checkpoint_data.get("log_entries", [])
            restored_count = 0
            
            for entry in log_entries:
                if entry.get("type") == "interaction" and "simple_mode" in entry.get("interaction_type", ""):
                    try:
                        # Extract state info from prompt
                        prompt = entry.get("prompt", "")
                        response = entry.get("response", "")
                        timestamp_str = entry.get("timestamp", "")
                        
                        # Parse coordinates from prompt
                        coords_match = re.search(r"Position: X=(\d+), Y=(\d+)", prompt)
                        coords = None
                        if coords_match:
                            coords = (int(coords_match.group(1)), int(coords_match.group(2)))
                        
                        # Parse context from prompt  
                        context = "overworld"  # default
                        if "Game State: battle" in prompt:
                            context = "battle"
                        elif "DIALOGUE:" in prompt or "dialogue" in prompt.lower():
                            context = "dialogue"
                        elif "menu" in prompt.lower():
                            context = "menu"
                        
                        # Extract action from response
                        action_taken = "UNKNOWN"
                        if "ACTION:" in response:
                            action_section = response.split("ACTION:")[-1].strip()
                            action_line = action_section.split('\n')[0].strip()
                            action_taken = action_line
                        
                        # Parse timestamp
                        timestamp = datetime.now()
                        if timestamp_str:
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str)
                            except:
                                pass
                        
                        # Create simplified game state summary
                        game_state_summary = f"Position: {coords}" if coords else "Position unknown"
                        if coords:
                            game_state_summary += f" | Context: {context}"
                        
                        # Add reasoning summary
                        reasoning = ""
                        if "REASONING:" in response:
                            reasoning_section = response.split("REASONING:")[-1].split("ACTION:")[0].strip()
                            reasoning = reasoning_section
                        
                        action_with_reasoning = f"{action_taken} | Reasoning: {reasoning}" if reasoning else action_taken
                        
                        # Create history entry
                        history_entry = HistoryEntry(
                            timestamp=timestamp,
                            player_coords=coords,
                            map_id=None,  # Not available in checkpoint
                            context=context,
                            action_taken=action_with_reasoning,
                            game_state_summary=game_state_summary
                        )
                        
                        self.state.history.append(history_entry)
                        
                        # Also add to recent actions if it's a valid action
                        if action_taken and action_taken not in ["UNKNOWN", "WAIT"]:
                            # Parse multiple actions if comma-separated
                            actions = [a.strip() for a in action_taken.replace(',', ' ').split()]
                            for action in actions:
                                if action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']:
                                    self.state.recent_actions.append(action)
                        
                        restored_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error parsing checkpoint entry: {e}")
                        continue
            
            # Update step counter to match checkpoint
            self.state.step_counter = restored_count
            
            logger.info(f"✅ Restored {restored_count} history entries from {checkpoint_file}")
            logger.info(f"   History: {len(self.state.history)} entries")
            logger.info(f"   Recent actions: {len(self.state.recent_actions)} actions")
            logger.info(f"   Step counter: {self.state.step_counter}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load history from checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_history_to_llm_checkpoint(self, checkpoint_file: str = None):
        """Save SimpleAgent history using LLM logger checkpoint system"""
        try:
            from utils.llm_logger import get_llm_logger
            
            # Get the global LLM logger instance
            llm_logger = get_llm_logger()
            if llm_logger is None:
                logger.warning("No LLM logger available for checkpoint saving")
                return False
            
            # Save checkpoint using LLM logger which includes cumulative metrics
            # The LLM logger will handle saving log_entries AND cumulative_metrics
            # If checkpoint_file is None, it will use the cache folder
            llm_logger.save_checkpoint(checkpoint_file, agent_step_count=self.state.step_counter)
            
            logger.info(f"💾 Saved LLM checkpoint to {checkpoint_file}")
            logger.info(f"   Step counter: {self.state.step_counter}")
            logger.info(f"   History: {len(self.state.history)} entries")
            logger.info(f"   Recent actions: {len(self.state.recent_actions)} actions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save LLM checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def record_failed_movement(self, coords: Tuple[int, int], direction: str, reason: str = "blocked"):
        """Record a failed movement attempt for future reference"""
        coord_key = f"{coords[0]},{coords[1]}"
        if coord_key not in self.state.failed_movements:
            self.state.failed_movements[coord_key] = []
        
        failed_entry = f"{direction}:{reason}"
        if failed_entry not in self.state.failed_movements[coord_key]:
            self.state.failed_movements[coord_key].append(failed_entry)
            logger.info(f"Recorded failed movement: {coord_key} -> {direction} ({reason})")
    
    def record_npc_interaction(self, coords: Tuple[int, int], interaction_type: str, notes: str = ""):
        """Record an NPC interaction for future reference"""
        coord_key = f"{coords[0]},{coords[1]}"
        interaction_info = f"{interaction_type}: {notes}" if notes else interaction_type
        self.state.npc_interactions[coord_key] = interaction_info
        logger.info(f"Recorded NPC interaction: {coord_key} -> {interaction_info}")
    
    def get_movement_memory(self, coords: Tuple[int, int]) -> str:
        """Get memory about failed movements and interactions at specific coordinates"""
        coord_key = f"{coords[0]},{coords[1]}"
        memory_parts = []
        
        # Check for failed movements
        if coord_key in self.state.failed_movements:
            failed_list = self.state.failed_movements[coord_key]
            memory_parts.append(f"Failed moves: {', '.join(failed_list)}")
        
        # Check for NPC interactions
        if coord_key in self.state.npc_interactions:
            interaction = self.state.npc_interactions[coord_key]
            memory_parts.append(f"NPC: {interaction}")
        
        return " | ".join(memory_parts) if memory_parts else ""
    
    def get_area_movement_memory(self, center_coords: Tuple[int, int], radius: int = 7) -> str:
        """Get movement memory for the area around the player"""
        cx, cy = center_coords
        memory_lines = []
        
        # Check nearby coordinates for failed movements or NPC interactions
        nearby_memories = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip current position
                
                check_coords = (cx + dx, cy + dy)
                memory = self.get_movement_memory(check_coords)
                if memory:
                    nearby_memories.append(f"({check_coords[0]},{check_coords[1]}): {memory}")
        
        if nearby_memories:
            memory_lines.append("🧠 MOVEMENT MEMORY (nearby area):")
            for memory in nearby_memories[:5]:  # Limit to 5 most relevant
                memory_lines.append(f"  {memory}")
        
        return "\n".join(memory_lines)
    
    def clear_movement_memory(self, partial: bool = False):
        """
        Clear movement memory (failed movements and NPC interactions).
        
        Args:
            partial: If True, only clear old entries (keep recent 5). If False, clear all.
        """
        if partial and (self.state.failed_movements or self.state.npc_interactions):
            # Keep only the 5 most recent entries for each
            if len(self.state.failed_movements) > 5:
                # Convert to list of tuples, sort by insertion order (dict maintains order in Python 3.7+)
                # Keep last 5 entries
                items = list(self.state.failed_movements.items())
                self.state.failed_movements = dict(items[-5:])
                logger.info(f"Partially cleared movement memory, kept {len(self.state.failed_movements)} recent failed movements")
            
            if len(self.state.npc_interactions) > 5:
                items = list(self.state.npc_interactions.items())
                self.state.npc_interactions = dict(items[-5:])
                logger.info(f"Partially cleared NPC interactions, kept {len(self.state.npc_interactions)} recent interactions")
        else:
            # Clear all movement memory
            cleared_movements = len(self.state.failed_movements)
            cleared_npcs = len(self.state.npc_interactions)
            self.state.failed_movements.clear()
            self.state.npc_interactions.clear()
            logger.info(f"Cleared all movement memory: {cleared_movements} failed movements, {cleared_npcs} NPC interactions")
        
        # Reset the action counter
        self.state.movement_memory_action_counter = 0
    
    def analyze_movement_preview(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the movement preview data from game state to find valid moves.
        
        Returns:
            Dict with 'walkable_directions', 'blocked_directions', and 'special_tiles'
        """
        walkable_directions = []
        blocked_directions = []
        special_tiles = {}
        
        # Look for movement preview in the formatted state
        formatted_state = format_state_for_llm(game_state)
        lines = formatted_state.split('\n')
        
        in_movement_preview = False
        for line in lines:
            if 'MOVEMENT PREVIEW:' in line:
                in_movement_preview = True
                continue
            
            if in_movement_preview:
                # Parse movement preview lines
                # Format: "  UP   : ( 15, 10) [.] WALKABLE - Optional description"
                if line.strip() and ':' in line:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        direction = parts[0].strip()
                        rest = parts[1].strip()
                        
                        if direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                            if 'WALKABLE' in rest:
                                walkable_directions.append(direction)
                                # Check for special tiles (check stairs before doors to avoid mislabeling)
                                if 'Stairs/Warp' in rest:
                                    special_tiles[direction] = 'stairs'
                                elif 'Door/Entrance' in rest:
                                    special_tiles[direction] = 'door'
                                elif 'Tall grass' in rest:
                                    special_tiles[direction] = 'grass'
                                elif 'Jump ledge' in rest and 'can jump' in rest:
                                    special_tiles[direction] = 'ledge'
                            elif 'BLOCKED' in rest:
                                blocked_directions.append(direction)
                elif not line.strip():
                    # Empty line typically ends the movement preview section
                    in_movement_preview = False
        
        return {
            'walkable_directions': walkable_directions,
            'blocked_directions': blocked_directions,
            'special_tiles': special_tiles
        }
    
    def validate_movement_sequence(self, movements: List[str], game_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if a sequence of movements is valid based on current state.
        
        Args:
            movements: List of movement directions
            game_state: Current game state
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not movements:
            return True, "No movements to validate"
        
        # Analyze current movement options
        movement_info = self.analyze_movement_preview(game_state)
        walkable = movement_info['walkable_directions']
        blocked = movement_info['blocked_directions']
        
        # Check first movement
        first_move = movements[0].upper()
        if first_move in blocked:
            return False, f"First movement {first_move} is BLOCKED"
        
        if first_move not in walkable and first_move in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return False, f"First movement {first_move} is not confirmed WALKABLE"
        
        # For multiple movements, only allow if we're very confident
        if len(movements) > 1:
            # We can't predict beyond the first move accurately
            # So we should discourage chaining unless explicitly safe
            return False, "Cannot validate multi-step movements - use single steps instead"
        
        return True, "Movement validated"

    def get_history_stats(self) -> Dict[str, int]:
        """Get current history tracking statistics"""
        return {
            "history_entries": len(self.state.history),
            "max_history_entries": self.state.history.maxlen,
            "recent_actions": len(self.state.recent_actions),
            "max_recent_actions": self.state.recent_actions.maxlen,
            "history_display_count": self.history_display_count,
            "actions_display_count": self.actions_display_count,
            "objectives_count": len(self.state.objectives),
            "step_counter": self.state.step_counter,
            "failed_movements": len(self.state.failed_movements),
            "npc_interactions": len(self.state.npc_interactions),
            "movement_memory_action_counter": self.state.movement_memory_action_counter,
            "movement_memory_clear_interval": self.movement_memory_clear_interval
        }

# Global simple agent instance for backward compatibility with existing multiprocess code
_global_simple_agent = None

def get_simple_agent(vlm) -> SimpleAgent:
    """Get or create the global simple agent instance"""
    global _global_simple_agent
    if _global_simple_agent is None:
        _global_simple_agent = SimpleAgent(vlm)
        
        # Check if we should load from checkpoint
        import os
        if os.environ.get("LOAD_CHECKPOINT_MODE") == "true":
            # Check cache folder first, then fall back to old location
            cache_dir = ".pokeagent_cache"
            checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt") if os.path.exists(cache_dir) else "checkpoint_llm.txt"
            if not os.path.exists(checkpoint_file) and os.path.exists("checkpoint_llm.txt"):
                checkpoint_file = "checkpoint_llm.txt"
            if os.path.exists(checkpoint_file):
                logger.info(f"🔄 Loading SimpleAgent history from {checkpoint_file}")
                _global_simple_agent.load_history_from_llm_checkpoint(checkpoint_file)
            else:
                logger.info(f"⚠️ No checkpoint file found: {checkpoint_file}")
                
    elif _global_simple_agent.vlm != vlm:
        # VLM changed, create new instance
        _global_simple_agent = SimpleAgent(vlm)
        
        # Load checkpoint for new instance too if mode is set
        import os
        if os.environ.get("LOAD_CHECKPOINT_MODE") == "true":
            # Check cache folder first, then fall back to old location
            cache_dir = ".pokeagent_cache"
            checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt") if os.path.exists(cache_dir) else "checkpoint_llm.txt"
            if not os.path.exists(checkpoint_file) and os.path.exists("checkpoint_llm.txt"):
                checkpoint_file = "checkpoint_llm.txt"
            if os.path.exists(checkpoint_file):
                logger.info(f"🔄 Loading SimpleAgent history from {checkpoint_file}")
                _global_simple_agent.load_history_from_llm_checkpoint(checkpoint_file)
                
    return _global_simple_agent

def simple_mode_processing_multiprocess(vlm, game_state, args=None):
    """Simple mode processing function for multiprocess mode (backward compatibility)"""
    # args parameter kept for backward compatibility but not used
    _ = args  # Acknowledge unused parameter
    agent = get_simple_agent(vlm)
    frame = game_state["visual"]["screenshot"]
    
    # CRITICAL: Validate frame before processing
    if frame is None:
        logger.error("🚫 CRITICAL: simple_step called with None frame")
        return "WAIT"
    
    return agent.process_step(frame, game_state)
