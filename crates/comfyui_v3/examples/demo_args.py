def demo(a, b, c, d):
    print(a, b, c, d)


def demo2(*args, **kwargs):
    print(args, kwargs)
    pass


def demo3(*args, **kwargs):
    demo(*args, **kwargs)
    pass

def demo4(a, b, **kwargs):
    demo(a, b, **kwargs)
    pass

def demo5(**kwargs):
    demo(**kwargs)
    demo2(**kwargs)
    pass

def main():
    demo(1, 2, 3, 4)
    demo2(1, 2, 3, 4)
    demo2(1, 2, c=3, d=4)
    demo3(1, 2, 3, 4)
    demo4(1, 2, c=3, d=4)
    demo5(a=1, b=2, c=3, d=4)
    # demo5(1, 2, 3, 4)


if __name__ == "__main__":
    main()
