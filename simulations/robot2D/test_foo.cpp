

#include <iostream>

using namespace std;

class Foo
{
	protected :

	Foo()
	{

	}

	public :

	static Foo* instance;

	static void destroy()
	{
		delete Foo::instance;
	}

	static void create()
	{
		cout << "created foo object" << endl;
		Foo::instance = new Foo();
	}

	virtual void hello()
	{
		cout << "hello from foo" << endl;
	}
};

Foo* Foo::instance = NULL;

class Bar : public Foo
{
	private :

	Bar() : Foo()
	{

	}

	public :

	static void create()
	{
		cout << "created bar object" << endl;
		Bar::instance = new Bar();
	}

	void hello()
	{
		cout << "hello from bar" << endl;
	}
};



int main()
{
	Bar::create();

	Bar::instance->hello();

	Bar::destroy();

	return 0;
}