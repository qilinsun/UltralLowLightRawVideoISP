#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

using namespace std;

int frame_length = 11319624;
void rawDecoder(string path, int width, int height)
{
    ifstream input(path, ifstream::binary);
    int raw_length = width * height;
    uint8_t buffer[3] = {0};
    unsigned short d0 = 0, d1 = 0;
    short high_mask = 0b11110000, low_mask = 0b00001111, raw_mask = 0b111111111111;
    short* raw_image;
    raw_image = (short *)malloc(sizeof(short)*10000000);
    int count = 0;
    for (int i = 0; i < frame_length; i+=3)
    {
        input.read((char *)buffer, 3);
        d0 = (unsigned short)buffer[0] << 4 | ((unsigned short)buffer[2] & low_mask);
        d1 = (unsigned short)buffer[1] << 4 | ((unsigned short)buffer[2] & high_mask); 
        raw_image[count++] = d0 & raw_mask;
        raw_image[count++] = d1 & raw_mask;
    }
    input.close();
    ofstream raw_test("raw_test.bin", ofstream::binary);
    raw_test.write((char *)raw_image, raw_length*2);
    raw_test.close();
}

void eventDecoder(string path, int width, int height)
{
    ifstream input(path, ifstream::binary);
    int event_length = width * height / 4;
    uint8_t tmp;
    unsigned char *event_image;
    event_image = (unsigned char *)malloc(sizeof(unsigned char)*10000000);
    uint8_t mask0 = 0b00000011;
    uint8_t mask1 = 0b00001100;
    uint8_t mask2 = 0b00110000;
    uint8_t mask3 = 0b11000000;
    int count = 0;
    for (int i = 0; i < event_length; i++)
    {
        input.read((char *)&tmp, 1);
        uint8_t p0 = tmp & mask0;
        uint8_t p1 = (tmp & mask1) >> 2;
        uint8_t p2 = (tmp & mask2) >> 4;
        uint8_t p3 = (tmp & mask3) >> 6;
        event_image[count++] = p0;
        event_image[count++] = p1;
        event_image[count++] = p2;
        event_image[count++] = p3;
    }
    input.close();
    ofstream output("event_test.bin", ofstream::binary);
    output.write((char *)event_image, event_length*4);
    output.close();
}


void eventDecoder(uint8_t tmp, int start, uint8_t *buffer)
{
    uint8_t mask0 = 0b00000011;
    uint8_t mask1 = 0b00001100;
    uint8_t mask2 = 0b00110000;
    uint8_t mask3 = 0b11000000;
    uint8_t* start_point = buffer + start;
    int count = 0;

    for (int i = 0; i < 4; i++)
    {
        uint8_t p0 = tmp & mask0;
        uint8_t p1 = (tmp & mask1) >> 2;
        uint8_t p2 = (tmp & mask2) >> 4;
        uint8_t p3 = (tmp & mask3) >> 6;
        *start_point = p0;
        *(start_point + 1) = p1;
        *(start_point + 2) = p2;
        *(start_point + 3) = p3;
    }
}


void eventStreamDecoder(string path, int frame_num, int offset, int width, int height){
    ifstream input(path, ifstream::binary);
    int frame_count = 0;
    int current_offset = 0;
    int event_length = width * height / 4;
    int count = 0;
    uint8_t tmp = 0;
    ofstream output("event_stream_test.bin", ofstream::binary|ofstream::app);
    for (int i = 0; i < frame_num; i++)
    {
        uint8_t *event_image;
        event_image = (uint8_t *)malloc(sizeof(unsigned char)*10000000);
        input.seekg(current_offset, ios::beg);
        current_offset += offset;
        for (int j = 0; j < event_length; j++)
        {
            input.read((char *)&tmp, 1);
            eventDecoder(tmp, count, event_image);
            count +=4;
        }
        output.write((char *)event_image, event_length*4);
    }
    input.close();
    output.close();
}


int main(){
    // FILE *f = fopen("bayer_bit12_3264_2312_20240909100046592.bin", "rb");
    // int width = 3264;
    // int height = 2312;
    // int frame_length = 11319624;
    // int raw_length = width * height;
    // uint8_t buffer[3];
    // unsigned short d0 = 0, d1 = 0;
    // short high_mask = 0b11110000, low_mask = 0b00001111, raw_mask = 0b111111111111;
    // short* raw_image;
    // raw_image = (short *)malloc(sizeof(short)*10000000);
    // int count = 0;
    // for (int i = 0; i < frame_length; i+=3)
    // {
    //     fread(buffer, sizeof(uint8_t), 3, f);
    //     d0 = (unsigned short)buffer[0] << 4 | ((unsigned short)buffer[2] & low_mask);
    //     d1 = (unsigned short)buffer[1] << 4 | ((unsigned short)buffer[2] & high_mask); 
    //     raw_image[count++] = d0 & raw_mask;
    //     raw_image[count++] = d1 & raw_mask;
    // }
    // fclose(f);471
    // cout << d0 << d1 << endl;
    // ofstream raw_test("raw_test.bin", ofstream::binary);
    // raw_test.write((char *)raw_image, raw_length*2);
    // raw_test.close();
    rawDecoder("bayer_bit12_3264_2312_20240909100046592.bin", 3264, 2312);
    eventDecoder("Normal2bit_1632_1156_20240909204933577.bin", 1632, 1156);
    eventStreamDecoder("Normal2bit_1632_1156_20240909204933577.bin", 10, 471720, 1632, 1156);
    return 0;
}