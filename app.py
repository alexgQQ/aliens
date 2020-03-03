import numpy as np
import pygame
import random
import argparse
import math
import time

from itertools import compress

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical


UP = pygame.USEREVENT + 1
DOWN = pygame.USEREVENT + 2
LEFT = pygame.USEREVENT + 3
RIGHT = pygame.USEREVENT + 4
UL = pygame.USEREVENT + 5
UR = pygame.USEREVENT + 6
DL = pygame.USEREVENT + 7
DR = pygame.USEREVENT + 8
ROT = pygame.USEREVENT + 9
WIDTH = 600
HEIGHT = 200


class MovableEntity(pygame.sprite.Sprite):
    """
    Base class to represent a moveable sprite.
    Override `get_action` before any `update` call to use movement functionality.
    """

    speed = 1
    action_list = (
        UP,
        DOWN,
        LEFT,
        RIGHT,
        UL,
        UR,
        DL,
        DR,
    )
    hit_boundary = False

    def check_boundary(self):
        if self.rect.x < 0:
            self.rect.x = 0
            self.hit_boundary = True
            self.action = None
        if self.rect.x > WIDTH - self.rect.size[0]:
            self.rect.x = WIDTH - self.rect.size[0]
            self.hit_boundary = True
            self.action = None
        if self.rect.y < 0:
            self.rect.y = 0
            self.hit_boundary = True
            self.action = None
        if self.rect.y > HEIGHT - self.rect.size[1]:
            self.rect.y = HEIGHT - self.rect.size[1]
            self.hit_boundary = True
            self.action = None

    def move_up(self):
        self.rect.move_ip(0, -1 * self.speed)
        self.check_boundary()

    def move_diag_ul(self):
        self.rect.move_ip(int(-0.707 * self.speed), int(-0.707 * self.speed))
        self.check_boundary()

    def move_diag_ur(self):
        self.rect.move_ip(int(0.707 * self.speed), int(-0.707 * self.speed))
        self.check_boundary()

    def move_down(self):
        self.rect.move_ip(0, 1 * self.speed)
        self.check_boundary()

    def move_diag_dl(self):
        self.rect.move_ip(int(-0.707 * self.speed), int(0.707 * self.speed))
        self.check_boundary()

    def move_diag_dr(self):
        self.rect.move_ip(int(0.707 * self.speed), int(0.707 * self.speed))
        self.check_boundary()

    def move_left(self):
        self.rect.move_ip(-1 * self.speed, 0)
        self.check_boundary()

    def move_right(self):
        self.rect.move_ip(1 * self.speed, 0)
        self.check_boundary()

    def get_action(self, *args, **kwargs):
        self.action = self.action_list[random.randint(0, len(self.action_list) - 1)]

    def update(self, *args, **kwargs):
        self.hit_boundary = False
        super().update(*args, **kwargs)
        self.get_action(*args, **kwargs)
        if self.action == UP:
            self.move_up()
        elif self.action == DOWN:
            self.move_down()
        elif self.action == LEFT:
            self.move_left()
        elif self.action == RIGHT:
            self.move_right()
        elif self.action == DL:
            self.move_diag_dl()
        elif self.action == DR:
            self.move_diag_dr()
        elif self.action == UL:
            self.move_diag_ul()
        elif self.action == UR:
            self.move_diag_ur()


class Background(pygame.sprite.Sprite):
    """
    Class to represent a scalable background image.
    """
    def __init__(self, size):
        super().__init__()
        image = pygame.image.load('img/background.jpg')
        self.image = pygame.transform.scale(image, size)
        self.rect = self.image.get_rect()
        self.rect.left = 0
        self.rect.top = 0


class Bullet(MovableEntity):
    """
    A Sprite that only moves quickly to the right, will remove Sprites it collides with in a target group.
    Will be removed from Groups if it hits a Sprite from a target Group or hits the boundary.
    """

    def __init__(self, center=None, target_group=None):
        super().__init__()
        self.speed = 10
        self.action = RIGHT
        self.image = pygame.image.load('img/bullet.png')
        self.rect = self.image.get_rect(center=center)
        self.target_group = target_group

    def get_action(self):
        pass

    def update(self):
        super().update()
        hit_enemies = pygame.sprite.spritecollide(self, self.target_group, True)
        if hit_enemies or self.hit_boundary:
            self.kill()


class Marine(MovableEntity):
    """
    This Sprite should only move up and down and spawn Bullets towards an Alien.
    It should be removed from Groups upon colliding with an Alien.
    """

    def __init__(self, alien_group=None, bullet_group=None):
        super().__init__()

        self.speed = 2
        self.dead = False
        self.action_list = (
            UP,
            DOWN,
        )

        self.image = pygame.image.load('img/marine.png')
        image_width, image_height = self.image.get_size()
        center = (
            image_width,
            random.randint(0 + image_height, HEIGHT - image_height),
        )
        self.rect = self.image.get_rect(center=center)
        self.alien_group = alien_group
        self.bullet_group = bullet_group
        self.start_time = time.time()
        self.get_action()

    @property
    def alien(self):
        try:
            return self.alien_group.sprites()[0]
        except IndexError:
            return None

    def get_action(self, *args, **kwargs):
        if self.alien is None:
            self.action = random.choice(self.action_list)
            return

        if self.alien.rect.y > self.rect.y:
            self.action = DOWN
        elif self.alien.rect.y < self.rect.y:
            self.action = UP

    def shoot(self):
        center = (
            self.rect.right,
            self.rect.centery,
        )
        bullet = Bullet(
            center=center,
            target_group=self.alien_group,
            )
        self.bullet_group.add(bullet)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

        if time.time() - self.start_time > 0.5:
            self.shoot()
            self.start_time = time.time()

    #     if self.dist_to_alien > 100 and self.dist_to_alien < 400:
    #         if self.aim():
    #             self.alien.dead = True

    # def aim(self):
    #     x = self.rect.centerx
    #     y = self.rect.centery
    #     alien_y = self.alien.rect.centery
    #     alien_x = self.alien.rect.centerx

    #     y_in_sights = alien_y - 25 < y and alien_y + 25 > y
    #     x_in_sights = alien_x - 25 < x and alien_x + 25 > x

    #     return y_in_sights or x_in_sights

    # @property
    # def dist_to_alien_x(self):
    #     return abs(self.alien.rect.centerx - self.rect.centerx)

    # @property
    # def dist_to_alien_y(self):
    #     return abs(self.alien.rect.centery - self.rect.centery)

    # @property
    # def dist_to_alien(self):
    #     return math.sqrt(self.dist_to_alien_x**2 + self.dist_to_alien_y**2)


class Alien(MovableEntity):
    """
    This Sprite can move freely in the game space and should be removed from Groups
    when colliding with a Bullet.
    """

    def __init__(self, enemy_group=None, bullet_group=None):
        super().__init__()

        self.has_eaten = False
        self.speed = 5
        self.dead = False

        self.image = pygame.image.load('img/alien.png')
        image_width, image_height = self.image.get_size()
        center = (
            WIDTH - image_width,
            random.randint(0 + image_height, HEIGHT - image_height),
        )
        self.rect = self.image.get_rect(center=center)
        self.enemy_group = enemy_group
        self.bullet_group = bullet_group

    @property
    def enemy(self):
        try:
            return self.enemy_group.sprites()[0]
        except IndexError:
            return None

    @property
    def dist_to_enemy_x(self):
        try:
            return abs(self.enemy.rect.centerx - self.rect.centerx)
        except AttributeError:
            return 0

    @property
    def dist_to_enemy_y(self):
        try:
            return abs(self.enemy.rect.centery - self.rect.centery)
        except AttributeError:
            return 0

    @property
    def dist_to_enemy(self):
        return math.sqrt(self.dist_to_enemy_x**2 + self.dist_to_enemy_y**2)

    @property
    def bullet(self):
        try:
            return self.bullet_group.sprites()[0]
        except IndexError:
            return None

    @property
    def dist_to_bullet_x(self):
        try:
            return abs(self.bullet.rect.centerx - self.rect.centerx)
        except AttributeError:
            return 0

    @property
    def dist_to_bullet_y(self):
        try:
            return abs(self.bullet.rect.centery - self.rect.centery)
        except AttributeError:
            return 0

    @property
    def dist_to_bullet(self):
        return math.sqrt(self.dist_to_bullet_x**2 + self.dist_to_bullet_y**2)

    @property
    def danger(self):
        def incoming(alien, bullet):
            return (
                bullet.rect.bottom <= alien.rect.bottom 
                and bullet.rect.bottom >= alien.rect.top
            )
        return any((incoming(self, bullet) for bullet in self.bullet_group.sprites()))

    def kill(self):
        self.dead = True
        super().kill()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

        if self.dead:
            self.image = pygame.image.load('img/dead.png')
            self.rect = self.image.get_rect(center=self.rect.center)


class TrainableAlien(Alien):
    """
    Extension of an Alien Sprite but with deep learning capabilities.
    """

    def __init__(self, preload=None, model=None, memory=[], epsilon=100, generation=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generation = generation
        self.gamma = 0.9
        self.learning_rate = 0.0005
        self.epsilon = epsilon - self.generation

        self.memory = []
        self.weight_file = 'weights.hdf5'

        self.preload = kwargs.get('preload')
        if model:
            self.model = model
        else:
            self.build_model(weights=self.preload)

    def remember(self):
        """
        Function to save state info for training as tuples.
        Should be called at the end of each update cycle.
        """
        self.memory.append(
            (self.initial_state, self.state, self.action_set, self.reward, self.dead))

    @property
    def state(self):
        """
        Property to represent states as an np.array of 0 and 1.
        """
        # enemy_x = self.rect.x if self.enemy is None else self.enemy.rect.x
        enemy_y = self.rect.y if self.enemy is None else self.enemy.rect.y
        state = [
            self.rect.y > enemy_y,
            self.rect.y < enemy_y,
            self.danger,
            # self.hit_boundary,
            # self.rect.x > self.enemy.rect.x,
            # self.rect.x < self.enemy.rect.x,
        ]
        return np.asarray(state, dtype=int)

    @property
    def num_of_states(self):
        return len(self.state)

    @property
    def num_of_actions(self):
        return len(self.action_list)

    @property
    def reward(self):
        """
        Property to return reward values for actions and the given state.
        """
        if self.has_eaten:
            return 20
        elif self.dead:
            return -20
        elif self.hit_boundary:
            return -5
        # reward = 0 if self.enemy_dx > 0 else self.enemy_dx
        scale = 1 if self.dist_to_bullet > 50 else -1
        reward = self.enemy_dx
        return reward * scale

    def get_action(self):
        """
        Return model predictions as actions
        """
        if self.preload and random.randint(0, 200) < self.epsilon:
            self.action_set = to_categorical(random.randint(0, self.num_of_actions - 1), num_classes=self.num_of_actions)
        else:
            prediction = self.model.predict(self.initial_state.reshape((1, self.num_of_states)))[0]
            self.action_set = to_categorical(np.argmax(prediction), num_classes=self.num_of_actions)
        self.action = list(compress(self.action_list, self.action_set))[0]

    def build_model(self, weights=None):
        """
        Initialize and save the neural network model.
        Input should be an array being the state of the agent.
        Ouput is an array representing the actions it should take.
        """
        model = Sequential()
        model.add(Dense(units=120, activation='relu', input_dim=self.num_of_states))
        model.add(Dropout(0.15))
        model.add(Dense(units=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(units=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(units=self.num_of_actions, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(self.weight_file)

        self.model = model

    def train_from_action(self):
        """
        Function to handle operations to fit a model from a single action
        """
        prediction = self.model.predict(self.state.reshape((1, self.num_of_states)))
        target = self.reward + self.gamma * np.amax(prediction)

        target_f = self.model.predict(self.initial_state.reshape((1, self.num_of_states)))
        target_f[0][np.argmax(self.action_set)] = target
        self.model.fit(self.initial_state.reshape((1, self.num_of_states)), target_f, epochs=1, verbose=0)

    def train_from_generation(self):
        """
        Function to replay memory and fit the model on past actions
        """
        if len(self.memory) > 1000:
            minibatch = random.sample(self.memory, 1000)
        else:
            minibatch = self.memory
        for prev_state, state, action, reward, dead in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([state]))[0])

            target_f = self.model.predict(np.array([prev_state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([prev_state]), target_f, epochs=1, verbose=0)

    def update(self, *args, **kwargs):

        # Decrease random action for exploration each update
        self.epsilon = 100 - self.generation

        # Get/set initial state values before action is taken
        self.has_eaten = False
        self.initial_state = self.state.copy()
        init_enemy_x = self.dist_to_enemy_x
        init_enemy_y = self.dist_to_enemy_y

        # Handle render and movement, set self.action prior to this
        super().update(*args, **kwargs)

        self.enemy_dx = init_enemy_x - self.dist_to_enemy_x
        self.enemy_dy = init_enemy_y - self.dist_to_enemy_y

        # Find enemies from groups that we hit
        eaten_enemies = pygame.sprite.spritecollide(self, self.enemy_group, True)
        if eaten_enemies:
            self.has_eaten = True
            # self.generation += 1

        self.remember()

        # if not self.preload:
        #     self.train_from_action()
        #     self.remember()
        #     if eaten_enemies or self.dead:
        #         self.train_from_cycle()

        # for enemy in eaten_enemies:
        #     enemy.kill()


class Game:
    def __init__(self, load=None, save=None):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.screen.fill((0, 0, 0))
        self.background = Background((WIDTH, HEIGHT))

        self.marine_group = pygame.sprite.Group()
        self.alien_group = pygame.sprite.Group()
        self.bullet_group = pygame.sprite.Group()
        self.alien = None

        self.load = load
        self.save = save

        self.stop = False
        self.fps = 30
        self.clock = pygame.time.Clock()
        self.clock.tick(self.fps)

    def init_round(self):
        self.marine_group.empty()
        self.alien_group.empty()
        self.bullet_group.empty()

        marine_kwargs = {
            'alien_group': self.alien_group,
            'bullet_group': self.bullet_group,
        }
        self.marine = Marine(**marine_kwargs)
        self.marine_group.add(self.marine)


        alien_kwargs = {
            'enemy_group': self.marine_group,
            'bullet_group': self.bullet_group,
            'preload': self.load,
        }
        if self.alien:
            alien_kwargs.update({
                'model': self.alien.model,
                'memory': self.alien.memory,
                'generation': self.alien.generation,
            })

        self.alien = TrainableAlien(**alien_kwargs)
        self.alien_group.add(self.alien)

    def update(self):

        self.screen.blit(self.background.image, self.background.rect)
        self.marine_group.update()
        self.bullet_group.update()
        self.alien.update()

        self.marine_group.draw(self.screen)
        self.bullet_group.draw(self.screen)
        self.alien_group.draw(self.screen)

        self.alien.train_from_action()

        if self.alien.dead or not bool(self.marine_group):
            # breakpoint()
            self.alien.train_from_generation()
            self.alien.generation += 1
            self.init_round()

    def run(self):
        self.clock.tick(self.fps)
        self.init_round()

        while not self.stop:
            # Look for an ESC keypress or quit signal to exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.stop = True

            self.screen.fill((0, 0, 0))
            self.update()
            pygame.display.flip()

        if self.save:
            self.alien.model.save_weights(self.alien.weight_file)


def parse_args():
    """
    Handle generic command arguments and return as a dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true', default=False,
                        help='Preload a weights file')
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help='Save a weights file')
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':

    args = parse_args()
    game = Game(load=args.get('load'), save=args.get('save'))
    game.run()
