######################################################################
######################################################################
####                                                              ####
#### One file version of git@github.com:alekoza02/CLI_tracker.git ####
####                                                              ####
######################################################################
######################################################################

class Color:
  def __init__(self, color_index=255):
    self.index = color_index


  def apply_color(self):
    return f"\033[38;5;{self.index}m"


  def set_rgb(self, R=5, G=5, B=5):
    '''Values range from 0 to 5.'''
    self.index = 16 + 36 * R + 6 * G + B
    return f"\033[38;5;{self.index}m"


  def set_index(self, index):
    self.index = index
    return f"\033[38;5;{self.index}m"


  def reset(self):
    return f"\033[0m"
  

import argparse

class FileManager:
  def __init__(self):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    
    self.args = parser.parse_args()
  
  
  def open_file(self):
    with open(self.args.file, "r") as f:
        content = f.read()
    return content

  
  def convert_to_array(self, content):
    lines = content.split("\n")
    result = []
    metadata = []

    for line in lines:
      try:
        coordinates = line.split()
        
        if len(coordinates) == 2:
          x = float(coordinates[0])
          y = float(coordinates[1])
          result.append((x, y))

        elif len(coordinates) == 3:
          x = float(coordinates[0])
          y = float(coordinates[1])
          z = float(coordinates[2])
          result.append((x, y, z))

        else:
          raise ValueError

      except ValueError:
        metadata.append(line)
    
    return result, metadata


  def load_data_from_file(self):
    c = self.open_file()
    data, metadata = self.convert_to_array(c)
    return data, metadata
  

import math

class PlotManager:
  
  def __init__(self, parent, color, second_color, round_y, measure_unit_y):
    self.parent = parent
    self.use_time_x = True
    self.use_time_y = False
    self.can_plot = False
    self.x_labels = []
    self.y_labels = []
    self.round_y = round_y
    self.measure_unit_y = measure_unit_y
    self.color = color
    self.second_color = second_color
    self.analytics = {
      'len' : None,
      'current_perc' : 0,
      'x_tracked' : None
    }

    self.set_x_limits()
    self.set_y_limits()
    self.set_x_unit('seconds')
    self.track_last_tail()


  def set_y_limits(self, lower=None, upper=None):
    self.y_lower_limit = lower
    self.y_upper_limit = upper
  
  
  def set_x_limits(self, lower=None, upper=None):
    self.x_lower_limit = lower
    self.x_upper_limit = upper


  def set_plot_data(self, data, metadata="", channels=1):
    if len(data) > 2 and not self.analytics['tracked_type'] is None:
      
      if self.analytics['tracked_type'] is 'elements':
        self.original = data[-self.analytics['x_tracked']:]
  
      elif self.analytics['tracked_type'] is 'seconds':
        x_last = data[-1][0]
        x_first = x_last - self.analytics['x_tracked']
        x_list = [coords[0] for coords in data]
        deltas = [abs(x - x_first) for x in x_list]
        index = deltas.index(min(deltas))
        self.original = data[index:]
        self.set_x_limits(max(0, x_first), x_last)
    
    else:
      self.original = data

    self.metadata = metadata
    self.can_plot = True

    self.analytics['len'] = len(data)
    
    if len(self.original) > 1:
      self.normalize_data(channels)
      self.apply_screen_coords(channels)
      self.get_x_labels()
      self.get_y_labels()
    else:
      self.can_plot = False
      self.parent.send_error("Insufficent points", self.x + int(self.w / 2), self.y + int(self.h / 2))


  def get_x_labels(self):
    delta = self.max_x - self.min_x

    values = [self.min_x + i * delta / 4 for i in range(5)]
    
    if self.use_time_x:
        if self.max_x < 60:
          self.x_labels = [f"{values[i]:.1f}s" for i in range(5)]
        elif self.max_x < 3600:
          self.x_labels = [f"{values[i] // 60:.0f}m {values[i] % 60:.0f}s" for i in range(5)]
        elif self.max_x < 3600 * 24:
          self.x_labels = [f"{values[i] // 3600:.0f}h {values[i] % 3600 // 60:.0f}m" for i in range(5)]
        else:
          self.x_labels = [f"{values[i] // (3600 * 24):.0f}d {values[i] % (3600 * 24) // 3600:.0f}h" for i in range(5)]
    
    else:
      self.x_labels = [f"{values[i]}" for i in range(5)]
  
  
  def get_y_labels(self):
    delta = self.max_y - self.min_y

    values = [self.min_y + i * delta / 4 for i in range(5)]
    
    if self.use_time_y:
        if self.max_y < 60:
          self.y_labels = [f"{values[i]:.1f}s" for i in range(5)]
        elif self.max_y < 3600:
          self.y_labels = [f"{values[i] // 60:.0f}m {values[i] % 60:.0f}s" for i in range(5)]
        elif self.max_y < 3600 * 24:
          self.y_labels = [f"{values[i] // 3600:.0f}h {values[i] % 3600 // 60:.0f}m" for i in range(5)]
        else:
          self.y_labels = [f"{values[i] // (3600 * 24):.0f}d {values[i] % (3600 * 24) // 3600:.0f}h" for i in range(5)]
    
    else:
      self.y_labels = [f"{values[i]}" for i in range(5)]
      

  def set_boundaries(self, x, y, w, h):
    self.x = x + 8
    self.y = y
    self.w = w - 2
    self.h = h - 2


  def normalize_data(self, channels=1):
    min_x, max_x, min_y, max_y = math.inf, -math.inf, math.inf, -math.inf
    self.normalized_data = []

    for coords in self.original:
      min_x = min(min_x, coords[0])
      max_x = max(max_x, coords[0])
      min_y = min(min_y, min([coords[i] for i in range(1, 2 * channels, 2)]))
      max_y = max(max_y, max([coords[i] for i in range(1, 2 * channels, 2)]))

    self.analytics['current_perc'] = max_y

    if not self.y_lower_limit is None:
      min_y = self.y_lower_limit
    
    if not self.y_upper_limit is None:
      max_y = self.y_upper_limit
    
    if not self.x_lower_limit is None:
      min_x = self.x_lower_limit
    
    if not self.x_upper_limit is None:
      max_x = self.x_upper_limit

    if min_y == max_y:
      min_y -= 1
      max_y += 1
    
    if min_x == max_x:
      min_x -= 1
      max_x += 1
    
    self.max_x, self.min_x, self.max_y, self.min_y = max_x, min_x, max_y, min_y

    delta_x, delta_y = max_x - min_x, max_y - min_y

    for coords in self.original:
      fin_coords = [(coords[0] - min_x) / (delta_x)]
      for i in range(1, 2 * channels, 2):
        fin_coords.extend([(coords[i] - min_y) / (delta_y), coords[i + 1]])
      
      self.normalized_data.append(tuple(fin_coords))


  def apply_screen_coords(self, channels=1):
    self.screen_data = []

    for coords in self.normalized_data:
      fin_coords = [self.x + (self.w - 6) * coords[0]]
      for i in range(1, 2 * channels, 2):
        fin_coords.extend([self.y + self.h * (1 - coords[i]), coords[i + 1]])
      
      self.screen_data.append(tuple(fin_coords))


  def set_x_unit(self, mode='seconds'):
    if mode == 'seconds':
      self.use_time_x = True

  
  def track_last_tail(self, elements=None, typology=None):
    '''Available modes: `seconds`, `elements`'''
    self.analytics['x_tracked'] = elements
    self.analytics['tracked_type'] = typology


import shutil
import math
import time

class Terminal:

  char_map = {
    "lu" : "╭",
    "ru" : "╮",
    "ld" : "╰",
    "rd" : "╯",
    "vert" : "│",
    "hori" : "─",
  }

  light_map = ".,-~*=!#@$"

  def __init__(self):
    self.update()
    self.reset_canvas()
    self.frame_number = 0
    self.finished = False
    self.main_color = Color() 
    self.file_manager = FileManager()  

    data_plot_settings = {
      "parent" : self,
      "color" : (3, 5, 5),
      "second_color" : (5, 5, 5),
      "round_y" : 0,
      "measure_unit_y" : "%",
    }

    interpol_plot_settings = {
      "parent" : self,
      "color" : (5, 3, 1),
      "second_color" : (5, 5, 2),
      "round_y" : 1,
      "measure_unit_y" : "s",
    }

    self.data_plot = PlotManager(**data_plot_settings)
    self.interpol_data = []
    self.interpol_plot = PlotManager(**interpol_plot_settings)

    self.start_frame_time = 0
    self.partial_frame_time = 1
    self.total_frame_time = 1


  def update(self):
    self.start_frame_time = time.perf_counter()
    self.w, self.h = self.get_size()
    self.aspect_ratio = self.w / self.h


  def get_size(self):
    return shutil.get_terminal_size()


  def reset(self, clear=False):
    if clear:
      print("\033[2J", end='', flush=True) 
    print("\033[H", end='', flush=True)


  def render_frame(self):
    self.reset()
    
    ris = ""
    for y in range(self.h):
      ris += "".join(self.canvas[y])
    print(ris, end='', flush=True)


  def render_plot(self, which_plot = 'data'):

    match which_plot:
      case 'data':
        plot = self.data_plot
      case 'interpol':
        plot = self.interpol_plot


    if plot.can_plot:
      
      # axis
      self.insert_string_vertical(self.char_map["vert"] * (plot.h + 1), plot.x - 1, plot.y, plot.color)
      self.insert_string_horizontal(self.char_map["hori"] * (plot.w - 4), plot.x - 1, plot.y + plot.h + 1, plot.color)
      
      # vertical ticks + labels
      self.insert_string_horizontal(f"{float(plot.y_labels[0]):>6.{plot.round_y}f}{plot.measure_unit_y}{Terminal.char_map["ld"]}", plot.x - 8, int(plot.y + 4 * (plot.h + 1) / 4), plot.color)
      self.insert_string_horizontal(f"{float(plot.y_labels[1]):>6.{plot.round_y}f}{plot.measure_unit_y}{Terminal.char_map["ru"]}", plot.x - 8, int(plot.y + 3 * (plot.h + 1) / 4), plot.color)
      self.insert_string_horizontal(f"{float(plot.y_labels[2]):>6.{plot.round_y}f}{plot.measure_unit_y}{Terminal.char_map["ru"]}", plot.x - 8, int(plot.y + 2 * (plot.h + 1) / 4), plot.color)
      self.insert_string_horizontal(f"{float(plot.y_labels[3]):>6.{plot.round_y}f}{plot.measure_unit_y}{Terminal.char_map["ru"]}", plot.x - 8, int(plot.y + 1 * (plot.h + 1) / 4), plot.color)
      self.insert_string_horizontal(f"{float(plot.y_labels[4]):>6.{plot.round_y}f}{plot.measure_unit_y}{Terminal.char_map["ru"]}", plot.x - 8, plot.y, plot.color)
      
      # horizontal labels
      self.insert_string_horizontal(plot.x_labels[0], + plot.x - 2, plot.y + plot.h + 2, plot.color)
      self.insert_string_horizontal(plot.x_labels[1], 1 - len(plot.x_labels[1]) + int(plot.x - 2 + 1 * (plot.w - 4) / 4), plot.y + plot.h + 2, plot.color)
      self.insert_string_horizontal(plot.x_labels[2], 1 - len(plot.x_labels[2]) + int(plot.x - 2 + 2 * (plot.w - 4) / 4), plot.y + plot.h + 2, plot.color)
      self.insert_string_horizontal(plot.x_labels[3], 1 - len(plot.x_labels[3]) + int(plot.x - 2 + 3 * (plot.w - 4) / 4), plot.y + plot.h + 2, plot.color)
      self.insert_string_horizontal(plot.x_labels[4], 1 - len(plot.x_labels[4]) + int(plot.x - 2 + 4 * (plot.w - 4) / 4), plot.y + plot.h + 2, plot.color)

      # horizontal ticks
      self.insert_string_horizontal(f"{Terminal.char_map["ru"]}", int(plot.x - 2 + 2), plot.y + plot.h + 1, plot.color)
      self.insert_string_horizontal(f"{Terminal.char_map["ru"]}", int(plot.x - 2 + 1 * (plot.w - 4) / 4), plot.y + plot.h + 1, plot.color)
      self.insert_string_horizontal(f"{Terminal.char_map["ru"]}", int(plot.x - 2 + 2 * (plot.w - 4) / 4), plot.y + plot.h + 1, plot.color)
      self.insert_string_horizontal(f"{Terminal.char_map["ru"]}", int(plot.x - 2 + 3 * (plot.w - 4) / 4), plot.y + plot.h + 1, plot.color)
      self.insert_string_horizontal(f"{Terminal.char_map["ru"]}", int(plot.x - 2 + 4 * (plot.w - 4) / 4), plot.y + plot.h + 1, plot.color)

      # More nuanced ASCII shading based on proximity to integer coordinates
      for coords in plot.screen_data:
        channels = (len(coords) - 1) // 2
        for channel in range(channels):
          self.insert_char(f"{self.main_color.set_rgb(*coords[2 + 2 * channel])}*", int(round(coords[0])), int(round(coords[1 + 2 * channel])))

          if which_plot == 'data':
            self.insert_string_vertical(f"." * (self.data_plot.h - int(round(coords[1])) + 1), int(round(coords[0])), int(round(coords[1])) + 1, coords[2 + 2 * channel])


    
  def send_error(self, message, x, y):
    x_pos = x - len(message) // 2
    self.insert_string_horizontal(message, x_pos, y, (5, 3, 3))


  def insert_char(self, char, x, y, color=(5, 5, 5)):
    try:
      self.canvas[y][x] = self.main_color.set_rgb(*color) + char + self.main_color.reset()
    except IndexError:
      ...


  def insert_string_horizontal(self, string, x, y, color=(5, 5, 5)):
    try:
      for index, char in enumerate(string):
        self.canvas[y][x + index] = self.main_color.set_rgb(*color) + char
      self.canvas[y][x + len(string) - 1] += self.main_color.reset()
    except IndexError:
      ...


  def insert_string_vertical(self, string, x, y, color=(5, 5, 5)):
    try:
      for index, char in enumerate(string):
        self.canvas[y + index][x] = self.main_color.set_rgb(*color) + char
      self.canvas[y + len(string) - 1][x] += self.main_color.reset()
    except IndexError:
      ...


  def reset_canvas(self):
    self.canvas = [[" " for i in range(self.w)] for j in range(self.h)]


  def insert_box(self, x, y, w, h, color=(5, 5, 5)):
    # rounded borders
    self.insert_char(self.char_map["lu"], x, y, color)
    self.insert_char(self.char_map["ld"], x, y + h, color)
    self.insert_char(self.char_map["rd"], x + w, y + h, color)
    self.insert_char(self.char_map["ru"], x + w, y, color)

    # quad edges
    self.insert_string_horizontal(self.char_map["hori"] * (w - 1), x + 1, y, color)
    self.insert_string_horizontal(self.char_map["hori"] * (w - 1), x + 1, y + h, color)
    self.insert_string_vertical(self.char_map["vert"] * (h - 1), x, y + 1, color)
    self.insert_string_vertical(self.char_map["vert"] * (h - 1), x + w, y + 1, color)
    

  def insert_colored_frame(self, x, y, w, h, text, color):
    self.insert_box(x, y, w, h, color)
    self.insert_string_horizontal(text, x + 3, y, color)


  def update_data_from_file(self):
    c, m = self.file_manager.load_data_from_file()
    c = self.color_plot(c, self.data_plot.color)
    self.data_plot.set_plot_data(c, m)
    
  
  def color_plot(self, c, color):
    import random
    for i, coords in enumerate(c):
      if len(coords) == 2:
        c[i] = tuple([*coords, color])
    return c


  def update_interpol(self):

    lunghezza_interpol = len(self.interpol_data)
    lunghezza_plot = len(self.data_plot.original)

    delta_elements = lunghezza_plot - lunghezza_interpol

    if delta_elements > 0 and len(self.data_plot.original) > 2 and not self.finished:

      x, y, yerr = [], [], []

      for index, coords in enumerate(self.data_plot.original[-min(max(50, int(len(self.data_plot.original) / 5)), len(self.data_plot.original)):]):
        x.append(coords[0])
        y.append(coords[1])
        yerr.append(1)      # no weight calculation


      # Weighted means
      w_sum = sum(yerr)
      xw_mean = sum(yerr[i] * x[i] for i in range(len(x))) / w_sum
      yw_mean = sum(yerr[i] * y[i] for i in range(len(y))) / w_sum

      # Weighted slope
      num = sum(yerr[i] * (x[i] - xw_mean) * (y[i] - yw_mean) for i in range(len(x)))
      den = sum(yerr[i] * (x[i] - xw_mean) ** 2 for i in range(len(x)))
      m = num / den
      b = yw_mean - m * xw_mean

      total_time = (100 - b) / m

      for i in range(delta_elements):
        self.interpol_data.append((self.data_plot.original[-1][0], total_time, self.interpol_plot.color, total_time - self.data_plot.original[-1][0], self.interpol_plot.second_color))

    elif delta_elements < -1 and len(self.data_plot.original) > 2:
      self.finished = False
      self.interpol_data = []

    # check for finished work
    try:
      if self.data_plot.original[-1][1] == 100 and not self.finished:
        self.finished = True
        self.interpol_data.append((self.data_plot.original[-1][0], self.data_plot.original[-1][0], self.interpol_plot.color, 0, self.interpol_plot.second_color))
    except IndexError:
      ...

    self.interpol_plot.set_plot_data(self.interpol_data, "", channels=2)


  def wait(self, seconds=1):
    time.sleep(seconds)


  def flip(self, fps=60):
    self.frame_number += 1
    self.render_frame()
    self.partial_frame_time = time.perf_counter() - self.start_frame_time
    self.wait(max(0, (1 / fps) - self.partial_frame_time))
    self.total_frame_time = time.perf_counter() - self.start_frame_time

  
  def get_fps(self):
    return self.partial_frame_time, self.total_frame_time
  
  
if __name__ == "__main__":
  
  print("\033[?25l")
  terminal = Terminal()
  terminal.interpol_plot.track_last_tail(300, 'seconds')
  
  try:
    while 1:

      terminal.data_plot.set_y_limits(0, 100)
      y_interpol = [min(coords[1], coords[3]) for coords in terminal.interpol_data]
      limite = min(y_interpol) if len(y_interpol) > 2 else 1
      terminal.interpol_plot.set_y_limits(min(0, limite), None)
      
      terminal.update()
      terminal.reset_canvas()

      x1, y1, x2, y2 = 1, 1, (terminal.w - 2) // 2, int(4 * terminal.h / 5) - 2
      terminal.data_plot.set_boundaries(x1, y1, x2 - x1, y2 - y1)
      
      x1, y1, x2, y2 = (terminal.w) // 2 + 2, 1, terminal.w - 3, int(4 * terminal.h / 5) - 2
      terminal.interpol_plot.set_boundaries(x1, y1, x2 - x1, y2 - y1)

      terminal.insert_colored_frame(terminal.w // 2 + 1, 0, terminal.w // 2 - 2, int(4 * terminal.h / 5) - 1, "ETA section", (5, 2, 1))
      terminal.insert_colored_frame(0, 0, terminal.w // 2, int(4 * terminal.h / 5) - 1, "Plot section", (2, 3, 4))
      terminal.insert_colored_frame(0, int(4 * terminal.h / 5), terminal.w - 1, int(terminal.h / 5), "Data section", (2, 5, 3))

      terminal.update_data_from_file()
      terminal.update_interpol()

      terminal.render_plot('data')
      terminal.render_plot('interpol')

      # name and lenght of the file
      terminal.insert_string_horizontal(f"Reporting '{terminal.file_manager.args.file}', with {terminal.data_plot.analytics["len"]} entries.", 2, int(4 * terminal.h / 5) + 2)
      
      # current progress
      terminal.insert_string_horizontal(f"{'Current progress:':<27} {terminal.data_plot.analytics['current_perc']:.1f}%", terminal.w // 2, int(4 * terminal.h / 5) + 2, terminal.data_plot.color)

      # interpolation data
      if len(terminal.interpol_data) > 2:
        terminal.insert_string_horizontal(f"{'TOTAL expected time:':<27} {terminal.interpol_data[-1][1]:.1f}s", terminal.w // 2, int(4 * terminal.h / 5) + 3, terminal.interpol_plot.color)
        terminal.insert_string_horizontal(f"{'ELAPSED time:':<27} {terminal.interpol_data[-1][0]:.1f}s", terminal.w // 2, int(4 * terminal.h / 5) + 4)
        terminal.insert_string_horizontal(f"{'ESTIMATED remaining time:':<27} {terminal.interpol_data[-1][3]:.1f}s", terminal.w // 2, int(4 * terminal.h / 5) + 5, terminal.interpol_plot.second_color)

      rendering_fps, capped_fps = terminal.get_fps()
      terminal.insert_string_horizontal(f"{'Rendering FPS:':<27} {1 / rendering_fps:.0f}", 2, int(4 * terminal.h / 5) + 3)
      terminal.insert_string_horizontal(f"{'Capped FPS:':<27} {1 / capped_fps:.0f}", 2, int(4 * terminal.h / 5) + 4)


      terminal.flip()
    
  except KeyboardInterrupt:
    terminal.reset(clear=True)
    print(terminal.main_color.reset())
