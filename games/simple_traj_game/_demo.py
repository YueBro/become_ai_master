import sys
sys.path.append("../..")
import os

from IPython import embed

import games.simple_traj_game as traj_game


IMG_BEFORE_ANSWERING = "../../outputs/img_before_ans.png"
IMG_AFTER_ANSWERING = "../../outputs/img_after_ans.png"
os.makedirs("../../outputs", exist_ok=True)


def main():
    game = traj_game.Game()

    attr: traj_game.Attributes = game.get_attributes()
    game.render_img(IMG_BEFORE_ANSWERING)
    ans = traj_game.Answer([(0.0, 0.0), (0.1, 0.0), (0.2, 0.1), (0.5, 0.5)])
    game.apply_answer(ans)
    game.render_img(IMG_AFTER_ANSWERING)
    ans_eval: traj_game.AnsEval = game.evaluate_ans()
    
    embed()


if __name__ == "__main__":
    main()
