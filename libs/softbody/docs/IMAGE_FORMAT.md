# Summary
The softbody library allows the state of a simulation to be serialized, so that it can be written to the disk or sent through the network.

Currently we don't actually save the complete simulation and a simulation reloaded from such a file is not guaranteed to be playable in the simulator. See notes below.

# Format
The overall structure of an image file resembles the [RIFF format](https://docs.microsoft.com/en-us/windows/win32/xaudio2/resource-interchange-file-format--riff-),
except that in chunk headers the size field is omitted. Since the data we usually serialize are containers of homogeneous values
and an individual chunk only contains a single container,
instead we store the count of elements.

All multibyte values are in little-endian order.

## Header
An image starts with a header:

```
 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|          Magic        |  Version  |   Flags   |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
```

Where magic is two FOURCCs: "`EASI`" "`sSIM`" (`49 53 41 45 4D 49 53 73`),
version is currently `1` and
all flag bits are reserved and should be constant zeroes.

## Arrays

For flat arrays of \<T> type, the chuck format is the following:
```
 0     1     2     3     4     5     6     7
+-----+-----+-----+-----+-----+-----+-----+-----+------...
| Four char identifier  |   Number of elements  | Data ...
+-----+-----+-----+-----+-----+-----+-----+-----+------...
```

## Maps

Maps from \<K> to \<V> are stored as if they were flat arrays of \<K, V> tuples:

```
 0     1     2     3     4     5     6     7
+-----+-----+-----+-----+-----+-----+-----+-----+-------...
| Four char identifier  |     Number of pairs   | Pairs ...
+-----+-----+-----+-----+-----+-----+-----+-----+-------...
```

## Maps to arrays

Maps from \<K> to arrays of \<V> are stored like this:

```
 0     1     2     3     4     5     6     7
+-----+-----+-----+-----+-----+-----+-----+-----+-------...
| Four char identifier  |     Number of pairs   | Pairs ...
+-----+-----+-----+-----+-----+-----+-----+-----+-------...
```

Where a single pair is encoded as:

```
 0          K_n+1 K_n+2 K_n+3 K_n+4
+----...---+-----+-----+-----+-----+------...
| Key...   |   Number of elements  | Data ...
+----...---+-----+-----+-----+-----+------...
```

## Extension data
Extension data is stored in it's own "master chunk". This master chunk may contain several subchucks. The header consists of only a chunk ID, whose value depends on which extension was active at serialization time.

```
 0     1     2     3
+-----+-----+-----+-----+------...
| Four char identifier  | Data ...
+-----+-----+-----+-----+------...
```

## List of FOURCCs
### Base
These chunks will always be present regardless of the active simulation extension.
* "`BPOS`" Array[Vec4]: Bind positions
* "`CPOS`" Array[Vec4]: Current positions
* "` VEL`" Array[Vec4]: Velocities
* "`AVEL`" Array[Vec4]: Angular velocities
* "`SIZE`" Array[Vec4]: Particle sizes
* "` QUA`" Array[Quat]: Particle orientations
* "`DENS`" Array[Float32]: Particle densities
* "`EDGE`" Map[u64, Array[u64]]: Particle-particle connections

### Extensions without saved data
While some extensions store no data in an image, every extension needs to have a master chunk identifier.
* "None": "`ExNo`"
* "Debug rope": "`ExDr`"
* "Debug cloth": "`ExDc`"

# Extension data: plant simulation
Master chunk identifier: "`ExPl`"

The master chunk begins with a header:
```
 0    1    2    3    4    5    6    7    8    9    10   11
+----+----+----+----+----+----+----+----+----+----+----+----+
|     Sentinel      |      Version      |  Number of chunks |
+----+----+----+----+----+----+----+----+----+----+----+----+
```
Where sentinel is the four char identifier "`PSsn`",
version is currently `0` and
chunk count in this version is at most `4`.

The header is then followed by these chunks:

* "`PSpa`" Map[u64, u64]: parents
* "`PSac`" Map[u64, u64]: apical child
* "`PSlb`" Map[u64, u64]: lateral bud
* "`PSap`" Map[u64, Vec4]: anchor points

# Notes

Now this file is called an image for a good reason:
the data stored in the file is not necessarily enough to continue the simulation, it's an imperfect image.
Simulation initial config for example is not written to the file.
Neither are cached values like bind pose centers of masses, nor the collision SDF.

The image save/load function's purpose was this:
provide a way for the users to save and reload the simulation,
so that they can regenerate the plant mesh with different meshgen parameters.

Now it may be possible that when you load an image, setup the colliders and press *Start*, the program continues the saved simulation without any errors.
Whether this happens depends on the code version, the image version and which extension you used when saving the simulation.

---
In later versions we should have a byte-size field in the chunk headers so that we can skip unknown chunks.