// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: serialization tests
//

#include "stdafx.h"
#include <catch2/catch.hpp>

#include "softbody.h"
#include "f_serialization.internal.h"

/*
Vector<float> serdes, Vector<Vec4> serdes, Map<index_t, Vec<index_t>> serdes, system serdes
*/

// TODO(danielm): these ser-des classes should be tested as well :^)

class Memory_Serializer : public sb::ISerializer {
public:
    Vector<uint8_t>&& get_buffer() {
        return std::move(buffer);
    }

    size_t write(void const* ptr, size_t size) override {
        if (write_ptr > buffer.size()) {
            return 0;
        }

        size_t ret = 0;

        size_t overwrite_point = buffer.size();
        auto p8 = (uint8_t const*)ptr;

        while (write_ptr < overwrite_point && size > 0) {
            buffer[write_ptr] = *p8;
            p8++;
            write_ptr++;
            ret++;
            size--;
        }

        while (size > 0) {
            buffer.push_back(*p8);
            p8++;
            write_ptr++;
            ret++;
            size--;
        }

        return ret;
    }

    void seek_to(size_t file_point) override {
        if (file_point > buffer.size()) {
            file_point = buffer.size();
        }

        write_ptr = file_point;
    }

    void seek(int offset) override {
        if (offset < 0) {
            while (offset < 0 && write_ptr > 0) {
                write_ptr--;
                offset++;
            }
        } else {
            while (offset > 0) {
                write_ptr++;
                offset--;
            }
        }
    }

    size_t tell() override {
        return write_ptr;
    }

private:
    Vector<uint8_t> buffer;
    size_t write_ptr = 0;
};

class Memory_Deserializer : public sb::IDeserializer {
public:
    Memory_Deserializer(Vector<uint8_t>&& buffer) : buffer(buffer), read_ptr(0) {
    }

    size_t read(void* ptr, size_t size) override {
        size_t end = buffer.size();
        size_t ret = 0;
        auto p8 = (uint8_t*)ptr;

        while (read_ptr < end && size > 0) {
            *p8 = buffer[read_ptr];
            p8++;
            read_ptr++;
            ret++;
            size--;
        }

        return ret;
    }

    void seek_to(size_t file_point) override {
        if (file_point > buffer.size()) {
            file_point = buffer.size();
        }

        read_ptr = file_point;
    }

    void seek(int offset) override {
        if (offset < 0) {
            while (offset < 0 && read_ptr > 0) {
                read_ptr--;
                offset++;
            }
        } else {
            while (offset > 0) {
                read_ptr++;
                offset--;
            }
        }
    }

    size_t tell() override {
        return read_ptr;
    }

private:
    Vector<uint8_t> buffer;
    size_t read_ptr;

};

#define CHUNK_TEST MAKE_4BYTE_ID('T', 'E', 'S', 'T')

TEST_CASE("Vector<float> serialization-deserialization") {
    Vector<float> test_data;
    for (int i = 0; i < 100; i++) {
        test_data.push_back((float)i);
    }

    Memory_Serializer ser;

    serialize(&ser, test_data, CHUNK_TEST);

    Memory_Deserializer deser(std::move(ser.get_buffer()));

    uint32_t id;
    deser.read(&id, sizeof(id));
    REQUIRE(id == CHUNK_TEST);

    Vector<float> deser_data;
    deserialize(&deser, deser_data);

    REQUIRE(test_data.size() == deser_data.size());
    
    for (size_t i = 0; i < test_data.size(); i++) {
        REQUIRE(test_data[i] == deser_data[i]);
    }
}

TEST_CASE("Vector<Vec3> serialization-deserialization") {
    Vector<Vec3> test_data;
    for (int i = 0; i < 100; i++) {
        test_data.push_back(Vec3((float)(i + 0), (float)(i + 1), (float)(i + 2)));
    }

    Memory_Serializer ser;

    serialize(&ser, test_data, CHUNK_TEST);

    Memory_Deserializer deser(std::move(ser.get_buffer()));

    uint32_t id;
    deser.read(&id, sizeof(id));
    REQUIRE(id == CHUNK_TEST);

    Vector<Vec3> deser_data;
    deserialize(&deser, deser_data);

    REQUIRE(test_data.size() == deser_data.size());
    
    for (size_t i = 0; i < test_data.size(); i++) {
        REQUIRE(test_data[i] == deser_data[i]);
    }
}
