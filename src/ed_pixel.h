/*
 * ed_pixel.h
 * Author: This file is courtesy of Team1 from CSS535, for your convenience
 *         Thanks to: Jack Phan, Anish Prasad, Mike Chavez
 * This file is the definition for a pixel
 * A pixel consists of three color channels: red, greed, and blue
 * This file also contains definitions for basic operators
 */
#ifndef _ED_PIXEL_H_
#define _ED_PIXEL_H_

#include <stdint.h>
typedef int8_t pixel_channel_t_signed;
#define pixel_channel_t uint8_t

struct pixel_t {
	pixel_channel_t red;
	pixel_channel_t green;
	pixel_channel_t blue;

	// Overloaded operators for comparing pixels
	// Only use red channel here

	// Equality operator
	bool operator ==(const pixel_t& rhs) {
		return (red == rhs.red);
	}

	// Inequality operator
	bool operator !=(const pixel_t& rhs) {
		return (red != rhs.red);
	}

	// Greater-than operator
	bool operator >(const pixel_t& rhs) {
		return (red > rhs.red);
	}

	// Greater-than or equal operator
	bool operator >=(const pixel_t& rhs) {
		return (red >= rhs.red);
	}

	// Less-than operator
	bool operator <(const pixel_t& rhs) {
		return (red < rhs.red);
	}

	// Less-than or equal operator
	bool operator <=(const pixel_t& rhs) {
		return (red <= rhs.red);
	}
};

// The same pixel with sign
struct pixel_t_signed {
	int8_t red;
	int8_t green;
	int8_t blue;
};

#endif // _ED_PIXEL_H_