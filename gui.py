import tkinter
from PIL import ImageGrab, Image, ImageTk
from backprop import predict

root = tkinter.Tk()
global canvas
canvas = tkinter.Canvas(root, width=400, height=400, bg='#000')
canvas2 = tkinter.Canvas(root, width=10, height=10, bg='#000')

# use best model
f = open('trained_networks/25_0.5_20')
network = eval(f.read()[:-1])
f.close()


def getter():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
    global image
    image = ImageGrab.grab().crop(tuple(map(lambda x: x * 2, (x, y, x1, y1))))
    # image.save('test.png')
    image.thumbnail((10, 10), Image.ANTIALIAS)
    image = image.convert('L')
    # image.show()
    # image.save('test_rsz.png')

    target = list(map(lambda x: x/255, list(image.getdata())))

    image = ImageTk.PhotoImage(image)
    canvas2.create_image(8, 8, image=image)

    print(predict(network, target))


submit = tkinter.Button(root, text='Submit', command=getter)
clear = tkinter.Button(root, text='Clear', command=lambda: [canvas.delete('all'), canvas2.delete('all')])


# when mouse is clicked and moving
def paint(event):
    x1, y1 = (event.x - 25), (event.y - 25)
    x2, y2 = (event.x + 25), (event.y + 25)
    canvas.create_oval(x1, y1, x2, y2, fill='#fff', outline='#fff')


canvas.bind('<B1-Motion>', paint)
canvas.bind('<Return>', getter)

canvas.pack()
canvas2.pack()
submit.pack()
clear.pack()

root.mainloop()
