"""
To keep memory overhead minimal, each of the functions in LIST_OF_QUEST_RESOLUTIONS will only return the idx of the winning player
Need to zero out the accumulated values if  the player id has not met the cs_id
"""
from functools import partial
import jax
import jax.numpy as jnp


def _culture_quest(game, cs_id):
    """
    Who produced the most culture?
    """
    # (6,), (6, 12)
    culture_accum = game.citystate_info.culture_tracker * game.have_met[:, 6:][:, cs_id]
    winning_player = culture_accum.argmax()
    return winning_player, (culture_accum > 0).sum() > 0 

def _faith_quest(game, cs_id):
    """
    Who produced the most faith?
    """
    faith_accum = game.citystate_info.faith_tracker * game.have_met[:, 6:][:, cs_id]
    winning_player = faith_accum.argmax()
    return winning_player, (faith_accum > 0).sum() > 0

def _techs_quest(game, cs_id):
    """
    Who researched the most techs?
    """
    current_techs_count = game.technologies.sum(-1)
    diff = current_techs_count - game.citystate_info.tech_tracker
    diff = diff * game.have_met[:, 6:][:, cs_id]
    winning_player = diff.argmax()
    return winning_player, (diff > 0).sum() > 0

def _traderoute_quest(game, cs_id):
    """
    Who traded the most?
    """
    trade_accum = game.citystate_info.trade_tracker[:, cs_id] * game.have_met[:, 6:][:, cs_id]
    winning_player = trade_accum.argmax()
    return winning_player, (trade_accum > 0).sum() > 0

def _religion_quest(game, cs_id):
    """
    Whose religion was the majority for the longest time?
    """
    religion_accum = game.citystate_info.religion_tracker * game.have_met[:, 6:][:, cs_id]
    winning_player = religion_accum.argmax()
    return winning_player, (religion_accum > 0).sum() > 0

def _wonder_quest(game, cs_id):
    """
    Who has the wonder for the longest?
    """
    wonder_accum = game.citystate_info.wonder_tracker * game.have_met[:, 6:][:, cs_id]
    winning_player = wonder_accum.argmax()
    return winning_player, (wonder_accum > 0).sum() > 0

def _resource_quest(game, cs_id):
    """
    Who had the resource for the longest?
    """
    resource_accum = game.citystate_info.resource_tracker * game.have_met[:, 6:][:, cs_id]
    winning_player = resource_accum.argmax()
    return winning_player, (resource_accum > 0).sum() > 0


LIST_OF_QUEST_RESOLUTIONS = [
    lambda x, y: (jnp.array(0), False), _culture_quest, _faith_quest, _techs_quest, _traderoute_quest, _religion_quest, _wonder_quest, _resource_quest
]

@partial(jax.vmap, in_axes=(None, 0, 0))
def resolve_quests(game, resolution_idx, cs_id):
    """
    If the quests need resolving  (to_resolve is True), then we take the result of the switch over LIST_OF_QUEST_RESOLUTIONS, otherwise no change
    """
    winner, was_a_winner = jax.lax.switch(
        resolution_idx.astype(jnp.int32),
        LIST_OF_QUEST_RESOLUTIONS,
        game,
        cs_id
    )
    return winner, was_a_winner
