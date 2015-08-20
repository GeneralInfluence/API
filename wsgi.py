#!/usr/bin/env python 2
"""nginx server"""

from api import app as application

if __name__ == "__main__":
    application.run()