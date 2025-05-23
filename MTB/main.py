# Copyright 2022 Kristof Floch
 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import sys
import os
from PyQt5.QtWidgets import QApplication, QSplashScreen
from gui import VideoWidget
from PyQt5.QtGui import QPixmap


def MotionTracker():
    # create app
    App = QApplication(sys.argv)


    # splash screen for loading
    splash = QSplashScreen(QPixmap(os.path.dirname(__file__)+"/images/logo.svg"))
    splash.show()
    App.processEvents()

    # open application
    root = VideoWidget()
    root.show()

    # close splash
    splash.finish(root)
    sys.exit(App.exec())


# run the application
if __name__ == "__main__":
    MotionTracker()