def demo(a, b, c, d):
    print(a, b, c, d)


def demo2(*args, **kwargs):
    print(args, kwargs)
    pass


def main():
    demo(1, 2, 3, 4)
    demo2(1, 2, 3, 4)
    demo2(1, 2, c=3, d=4)
    pass


if __name__ == "__main__":
    main()
