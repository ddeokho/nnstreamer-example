import os
import sys
import logging
import gi
import cairo

gi.require_version("Gst", "1.0")
gi.require_foreign('cairo')
gi.require_version('Gtk', '3.0')
from gi.repository import Gst, GObject


class NNStreamerExample : 
    """NNStreamer example for object dectection"""
 
    def __init__(self, argv=None):
        self.loop = None
        self.pipeline = None
        self.running = False

        self.current_label_index = -1
        self.new_label_index = -1

        #model
        self.tf_model = ""
        self.tf_labels = []

        #output names
        self.num_detections = 0.0
        self.detection_classes = 0.0
        self.detection_scores = 0.0
        self.detection_boxes = 0.0

        #drawing
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.class_id = 0
        self.prob = 0.0
    
        if not self.tf_init():
            raise Exception

        GObject.threads_init()
        Gst.init(argv)
    
    def run_example(self):
        """Init pipline and run example 
        
        :return: None
        """

        # main loop
        self.loop = GObject.MainLoop()

        # init pipline
        self.pipeline = Gst.parse_launch( 
             "v4l2src name=src ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! tee name=t_raw "
            "t_raw. ! queue ! videoconvert ! cairooverlay name=tensor_res ! ximagesink name=img_tensor "
            "t_raw. ! queue leaky=2 max-size-buffers=2 ! videoscale ! tensor_converter ! "
            "tensor_filter framework=tensorflow model="+self.tf_model+
            " ! tensor_sink name=tensor_sink "   
        )

        #"t_raw. ! queue leaky=2 max-size-buffers=2 ! videoscale ! tensor_converter ! "
            #"tensor_filter framework=tensorflow model="+self.tf_model+ 
            #"tensor_sink name=tensor_sink "
        #"input=3:640:480:1 inputname=image_tensor inputtype=uint8 "
        #"output=1,100:1,100:1,4:100:1 "
        #"outputname=num_detections,detection_classes,detection_scores,detection_boxes "
        #"outputtype=float32,float32,float32,float32 ! "
       

        #bus and message callback
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

        #tensor sink signal : new data callbaack
        tensor_sink = self.pipeline.get_by_name('tensor_sink')
        tensor_sink.connect('new-data', self.new_data_cb)

        #cario overlay
        cairo_overlay = self.pipeline.get_by_name('tensor_res')
        cairo_overlay.connect('draw', self.draw_overlay_cb)
        #cario_overlay.connect('caps-change', self.prepare_overlay_cb)


        #timer to update result
        GObject.timeout_add(500, self.on_timer_update_result)


        #start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True


        #Set window title
        self.set_window_title("img_tensor","NNStreamer Example")

        #run main loop
        self.loop.run()

        #quit when received eos or error message
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)


        bus.remove_signal_watch()


    def on_bus_message(self, bus, message):
        """Callback for message.

        :param bus: pipeline bus
        "param message: message from pipline
        :return : None
        """
        if message.type == Gst.MessageType.EOS:
            logging.info('received eos message')
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logging.warning('[error] %s : %s', error.message, debug)
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            error, debug = message.parse_warning()
            logging.warning('[warning] %s : %s',error.message, debug)
            
        elif message.type == Gst.MessageType.STREAM_START:
            logging.info('recesived start message')

        elif message.type == Gst.MessageType.QOS:
            data_format, processed, dropped = message.parse_qos_stats()
            format_str = Gst.Format.get_name(data_format)
            logging.debug('[qos] format[%s] processed[%d] dropped[%d]', format_str, processed, dropped)


    def draw_overlay_cb(self, _overlay, context, _timestamp, _duration, user_data):
        context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(40)
        context.move_to(100,100)
        context.text_path('HELLO')
        context.set_source_rgb(0.5, 0.5, 1)
        context.fill_preserve()
        context.set_source_rgb(0,0,0)
        context.set_line_width(1)
        context.stroke()


    
    def new_data_cb(self, buffer):
       
        #num_detections
        #mem_num = buffer.get_memory()
        #result, info_num = mem_num.map(Gst.MapFlag.READ)
        #if(info_num.size != 4):
        #    self.num_detections=info_num.data
       
        #detection_classes
        #mem_classes = gst

        if self.running:
            for idx in range(buffer.n_memory()):
                mem = buffer.peek_memory(idx)
                result, mapinfo = mem.map(Gst.MapFlag.READ)
                if result:
                    self.update_top_label_index(mapinfo.data, mapinfo.size)
                    mem.unmap(mapinfo)

    #def on_timer_update_result(self):
    #    if self.running:
    #        if self.current_label_index !=
    
    

    def set_window_title(self, name, title):
        """Set window title.

        :param name: GstXImageSink element name
        :param title: window title
        :return: None
        """
        element = self.pipeline.get_by_name(name)
        if element is not None:
            pad = element.get_static_pad('sink')
            if pad is not None:
                tags = Gst.TagList.new_empty()
                tags.add_value(Gst.TagMergeMode.APPEND, 'title', title)
                pad.send_event(Gst.Event.new_tag(tags))
            

    def tf_init(self):
        """Check tf model and load labels.

        :return: True if successfully initialized
        """

        tf_model = "ssdlite_mobilenet_v2.pb"
        tf_label = "coco_labels_list.txt"
        current_folder = os.path.dirname(os.path.abspath(__file__))
        model_folder = os.path.join(current_folder,"tf_model")


        #check model file exists
        self.tf_model = os.path.join(model_folder,tf_model)
        if not os.path.exists(self.tf_model):
            logging.error('cannot find tf model [%s]', self.tf_model)
            return False
        
        #load labels
        label_path = os.path.join(model_folder,tf_label)
        try:
            with open(label_path,'r') as label_file:
                for line in label_file.readlines():
                    self.tf_labels.append(line)

        except FileNotFoundError:
            logging.error('cannot find tf label [%s]',label_path)
            return False
        
        logging.info('finished to load labels, total [%d]',len(self.tf_labels))
        return True


    def update_top_label_index(self, data, data_size):
        """Update tf label index with max score.

        :param data: array of scores
        :param data_size: data size
        :return None
        """

        #-1 if failed to get max score index
        self.new_label_index = -1

        if data_size == len(self.tf_labels):
            scores = [data[i] for i in range(data_size)]
            max_score = max(scores)
            if max_score > 0:
                self.new_label_index = scores.index(max_score)
        else:
            logging.error('unexpected data size [%d]',data_size)


if __name__ == '__main__':
    example = NNStreamerExample(sys.argv[1:])
    example.run_example()
