BUILDDIR=build

CC=g++
CXXFLAGS+=-std=c++1y -Wall -fPIC
LDLIBS+=-lboost_program_options
LDLIBS+=-lopencv_core -lopencv_highgui -lopencv_imgproc

${BUILDDIR}/docscanner: ${BUILDDIR}/main.o
	${CC} ${CXXFLAGS} -o $@ $< ${LDLIBS}

${BUILDDIR}/main.o: src/main.cpp builddir
	${CC} ${CXXFLAGS} -c -o $@ $<

.PHONY: builddir
builddir:
	mkdir -p ${BUILDDIR}

.PHONY: clean
clean:
	rm -rf ${BUILDDIR}
