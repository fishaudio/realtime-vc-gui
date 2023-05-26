import sys
from PyQt6 import QtWidgets
from .gui import MainWindow

def main():
    # create the application and the main window
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    # setup stylesheet
    # apply_stylesheet(app, theme='dark_teal.xml')

    # run
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
