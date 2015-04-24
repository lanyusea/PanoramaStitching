panorama:	panorama.cpp
	g++ `pkg-config --cflags opencv` panorama.cpp -ggdb -o panorama `pkg-config --libs opencv`

clean:
	rm panorama
