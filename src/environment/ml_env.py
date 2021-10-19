import numpy as np


class OfflineEnv(object):
    def __init__(
        self,
        users_dict,
        users_dict_positive_items,
        users_history_lens,
        movies_groups,
        state_size,
        fairness_constraints,
        fix_user_id=None,
        reward_model=None,
    ):

        # users: interacted items, rate
        self.users_dict = users_dict if reward_model else users_dict_positive_items

        # users history length
        self.users_history_lens = users_history_lens

        self.state_size = state_size

        # filter users with len_history > state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = (
            fix_user_id if fix_user_id else np.random.choice(self.available_users)
        )
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][: self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000

        self.movies_groups = movies_groups
        self.group_count = {}
        self.total_recommended_items = 0
        self.fairness_constraints = fairness_constraints

        self.reward_model = reward_model

    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users

    def reset(self):
        self.user = (
            self.fix_user_id
            if self.fix_user_id
            else np.random.choice(self.available_users)
        )
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][: self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.group_count.clear()
        self.total_recommended_items = 0
        return self.user, self.items, self.done

    def step(self, action, top_k=False):

        reward = -1

        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                group = self.movies_groups[act]
                if group not in self.group_count:
                    self.group_count[group] = 0
                self.group_count[group] += 1
                self.total_recommended_items += 1

                if act in self.user_items.keys() and act not in self.recommended_items:
                    _reward = 0.5 * (self.user_items[act] - 3)
                    rewards.append(_reward)

                    if _reward > 0:
                        correctly_recommended.append(act)

                elif (
                    act not in self.user_items.keys()
                    and act not in self.recommended_items
                ):
                    if self.reward_model:
                        # TODO model.predict
                        _reward = (
                            self.reward_model.predict(
                                torch.tensor([self.user]).long().to("cuda"),
                                torch.tensor([act]).long().to("cuda"),
                            )
                            .cpu()
                            .data[0]
                        )
                        if _reward > 0:
                            correctly_recommended.append(act)
                    else:
                        rewards.append(0)

                else:
                    rewards.append(-1)

                self.recommended_items.add(act)

            if max(rewards) > 0:
                self.items = (
                    self.items[len(correctly_recommended) :] + correctly_recommended
                )
                self.items = self.items[-self.state_size :]
            reward = rewards

        else:
            group = self.movies_groups[action]
            if group not in self.group_count:
                self.group_count[group] = 0
            self.group_count[group] += 1
            self.total_recommended_items += 1

            if (
                action in self.user_items.keys()
                and action not in self.recommended_items
            ):
                reward = 0.5 * (self.user_items[action] - 3)

            elif (
                action not in self.user_items.keys()
                and action not in self.recommended_items
            ):
                if self.reward_model:
                    # TODO model.predict
                    reward = (
                        self.reward_model.predict(
                            torch.tensor([self.user]).long().to("cuda"),
                            torch.tensor([act]).long().to("cuda"),
                        )
                        .cpu()
                        .data[0]
                    )
                else:
                    reward = 0
            else:
                reward = -1

            if reward > 0:
                self.items = self.items[1:] + [action]
            self.recommended_items.add(action)

        if (
            len(self.recommended_items) > self.done_count
            or len(self.recommended_items)
            >= self.users_history_lens[list(self.users_dict.keys()).index(self.user)]
        ):
            self.done = True

        return self.items, reward, self.done, self.recommended_items


class OfflineFairEnv(OfflineEnv):
    def __init__(
        self,
        users_dict,
        users_dict_positive_items,
        users_history_lens,
        movies_groups,
        state_size,
        fairness_constraints,
        fix_user_id=None,
        reward_model=None,
    ):
        super().__init__(
            users_dict,
            users_dict_positive_items,
            users_history_lens,
            movies_groups,
            state_size,
            fairness_constraints,
            fix_user_id,
            reward_model,
        )

    def step(self, action, top_k=False):

        reward = -1

        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:

                group = self.movies_groups[act]
                if group not in self.group_count:
                    self.group_count[group] = 0
                self.group_count[group] += 1
                self.total_recommended_items += 1

                if act in self.user_items.keys() and act not in self.recommended_items:
                    _reward = 0.5 * (self.user_items[act] - 3)
                    if _reward > 0:
                        correctly_recommended.append(act)
                        rewards.append(
                            (
                                self.fairness_constraints[group - 1]
                                / sum(self.fairness_constraints)
                            )
                            - (self.group_count[group] / sum(self.group_count.values()))
                            + 1
                        )
                    else:
                        rewards.append(-1)

                elif (
                    act not in self.user_items.keys()
                    and act not in self.recommended_items
                ):
                    if self.reward_model:
                        # TODO model.predict
                        _reward = (
                            self.reward_model.predict(
                                torch.tensor([self.user]).long().to("cuda"),
                                torch.tensor([act]).long().to("cuda"),
                            )
                            .cpu()
                            .data[0]
                        )
                        if _reward > 0:
                            correctly_recommended.append(act)
                            rewards.append(
                                (
                                    self.fairness_constraints[group - 1]
                                    / sum(self.fairness_constraints)
                                )
                                - (
                                    self.group_count[group]
                                    / sum(self.group_count.values())
                                )
                                + 1
                            )
                        else:
                            rewards.append(-1)
                    else:
                        rewards.append(0)
                else:
                    rewards.append(-1)

                self.recommended_items.add(act)

            if max(rewards) > 0:
                self.items = (
                    self.items[len(correctly_recommended) :] + correctly_recommended
                )
                self.items = self.items[-self.state_size :]
            reward = rewards

        else:
            group = self.movies_groups[action]
            if group not in self.group_count:
                self.group_count[group] = 0
            self.group_count[group] += 1
            self.total_recommended_items += 1

            if (
                action in self.user_items.keys()
                and action not in self.recommended_items
            ):
                _reward = 0.5 * (self.user_items[action] - 3)
                if _reward > 0:
                    self.items = self.items[1:] + [action]
                    reward = (
                        (
                            self.fairness_constraints[group - 1]
                            / sum(self.fairness_constraints)
                        )
                        - (self.group_count[group] / sum(self.group_count.values()))
                        + 1
                    )
                else:
                    reward = -1
            elif (
                action not in self.user_items.keys()
                and action not in self.recommended_items
            ):
                if self.reward_model:
                    # TODO model.predict
                    _reward = (
                        self.reward_model.predict(
                            torch.tensor([self.user]).long().to("cuda"),
                            torch.tensor([action]).long().to("cuda"),
                        )
                        .cpu()
                        .data[0]
                    )
                    if _reward > 0:
                        self.items = self.items[1:] + [action]
                        reward = (
                            (
                                self.fairness_constraints[group - 1]
                                / sum(self.fairness_constraints)
                            )
                            - (self.group_count[group] / sum(self.group_count.values()))
                            + 1
                        )
                    else:
                        reward = -1
                else:
                    reward = 0
            else:
                reward = -1
            self.recommended_items.add(action)

        if (
            len(self.recommended_items) > self.done_count
            or len(self.recommended_items)
            >= self.users_history_lens[list(self.users_dict.keys()).index(self.user)]
        ):
            self.done = True

        return self.items, reward, self.done, self.recommended_items
