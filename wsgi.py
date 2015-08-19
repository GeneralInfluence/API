#!/usr/bin/env python 2
"""nginx server"""

import os, sys, inspect

this_dir = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
parent_dir = os.path.realpath(os.path.abspath(os.path.join(this_dir,"../")))
sys.path.insert(0, this_dir)
sys.path.insert(0, parent_dir)

from api import app

if __name__ == "__main__":
    app.run()