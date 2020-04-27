import tkinter
import turtle

sc = tkinter.Tk()
sc.geometry("1000x1000+100+100")

fr4 = tkinter.Frame(sc, height=500, width=600, bd=4, bg="light green", takefocus="", relief=tkinter.SUNKEN)

fr4.grid(row=2, column=2, sticky=(tkinter.N, tkinter.E, tkinter.W, tkinter.S))

# Canvas
canvas = tkinter.Canvas(fr4, width=750, height=750)
canvas.pack()

# Turtle
turtle1 = turtle.RawTurtle(canvas)
turtle1.color("black")
turtle1.shape("circle")


def drag_handler(x, y):
    turtle1.ondrag(None)  # disable event inside event handler
    turtle1.goto(x, y)
    print(x,y)
    turtle1.ondrag(drag_handler)  # reenable event on event handler exit

turtle1.ondrag(drag_handler)

sc.mainloop()