CXX  = -g++
CXXFLAGS = -std=c++11 -g
CPPFLAGS = -I/usr/local/include/opencv4
EXEC = ./main 
OBJFILES = task3.o HelperFunctions.o HOGDescriptor.o RandomForest.o Window.o
TARGET = main
LDFLAGS = -L/usr/local/lib/ -lopencv_ml -lopencv_objdetect -lopencv_optflow -lopencv_phase_unwrapping -lopencv_photo -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_calib3d -lopencv_ccalib -lopencv_core -lopencv_datasets -lopencv_dnn_objdetect -lopencv_dnn -lopencv_dpm -lopencv_face -lopencv_features2d -lopencv_flann -lopencv_freetype -lopencv_fuzzy -lopencv_gapi -lopencv_hfs -lopencv_highgui -lopencv_imgcodecs -lopencv_img_hash -lopencv_imgproc -lopencv_line_descriptor -lopencv_plot -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_stitching -lopencv_structured_light -lopencv_superres -lopencv_surface_matching -lopencv_text -lopencv_tracking -lopencv_videoio -lopencv_video -lopencv_videostab -lopencv_xfeatures2d -lopencv_ximgproc -lopencv_xobjdetect -lopencv_xphoto

all: $(TARGET)

$(TARGET): $(OBJFILES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJFILES) $(CPPFLAGS) $(LDFLAGS)

task3.o : task3.cpp  HOGDescriptor.h RandomForest.h HelperFunctions.h Window.h
	$(CXX) -c task3.cpp $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS)

helpfunc.o : HelperFunctions.cpp HelperFunctions.h
	$(CXX) -c HelperFunctions.cpp $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) 

hogdesc.o : HOGDescriptor.cpp HOGDescriptor.h
	$(CXX) -c HOGDescriptor.cpp $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) 

randomfor.o : RandomForest.cpp RandomForest.h TrainFilesPaths.h HOGDescriptor.h
	$(CXX) -c RandomForest.cpp $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) 

window.o : Window.cpp Window.h
	$(CXX) -c Window.cpp $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) 

clean:
	rm -f $(OBJFILES) $(TARGET)

run: all
	$(EXEC)



