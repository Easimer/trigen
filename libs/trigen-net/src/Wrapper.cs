using System;
using System.Runtime.InteropServices;

namespace Net.Easimer.Trigen
{
    public enum Status
    {
        OK = 0,
        Failure,
        InvalidArguments,
        OutOfMemory,
        InvalidConfiguration,
        InvalidMesh,
        NotReady,
        NotEnoughSpace,
        FunctionIsUnavailable,
    }

    [Flags]
    public enum Flags
    {
        None = 0,
        PreferCPU = 1 << 0,
        UseGeneralTexturingAPI = 1 << 1,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Parameters
    {
        public uint flags;

        [MarshalAs(UnmanagedType.ByValArray, ArraySubType = UnmanagedType.R4, SizeConst = 3)]
        public float[] seed_position;

        public float density;
        public float attachment_strength;
        public float surface_adaption_strength;
        public float stiffness;
        public float aging_rate;
        public float phototropism_response_strength;
        public float branching_probability;
        public float branch_angle_variance;

        public uint particle_count_limit;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ColliderMesh
    {
        public UIntPtr triangle_count;

        public IntPtr vertex_indices;
        public IntPtr normal_indices;

        public UIntPtr position_count;
        public IntPtr positions;

        public UIntPtr normal_count;
        public IntPtr normals;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Mesh
    {
        public UIntPtr triangle_count;

        public IntPtr vertex_indices;
        public IntPtr normal_indices;

        public UIntPtr position_count;
        public IntPtr positions;
        public IntPtr uvs;

        public UIntPtr normal_count;
        public IntPtr normals;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Transform
    {
        [MarshalAs(UnmanagedType.ByValArray, ArraySubType = UnmanagedType.R4, SizeConst = 3)]
        public float[] position;

        [MarshalAs(UnmanagedType.ByValArray, ArraySubType = UnmanagedType.R4, SizeConst = 4)]
        public float[] orientation;

        [MarshalAs(UnmanagedType.ByValArray, ArraySubType = UnmanagedType.R4, SizeConst = 3)]
        public float[] scale;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Texture
    {
        public IntPtr image;
        public UInt32 width;
        public UInt32 height;
    }

    public class Exception : System.Exception
    {
        public Status StatusCode { get; protected set; }

        public Exception(Status statusCode)
        {
            StatusCode = statusCode;
        }
    }

    public class Session : IDisposable
    {
        private IntPtr sessionHandle;

        protected Session(IntPtr sessionHandle)
        {
            this.sessionHandle = sessionHandle;
        }

        public void Dispose()
        {
            if(sessionHandle != IntPtr.Zero)
            {
                DestroySession(sessionHandle);
                sessionHandle = IntPtr.Zero;
            }
        }

        public static Session Create(Parameters param)
        {
            IntPtr handle;
            var status = CreateSession(out handle, ref param);
            if(status != Status.OK)
            {
                throw new Exception(status);
            }

            return new Session(handle);
        }

        [DllImport("trigen.dll", EntryPoint = "Trigen_CreateSession")]
        private static extern Status CreateSession([Out] out IntPtr handle, [In] ref Parameters param);

        [DllImport("trigen.dll", EntryPoint = "Trigen_DestroySession")]
        private static extern Status DestroySession([In] IntPtr handle);

        [DllImport("trigen.dll", EntryPoint = "Trigen_CreateCollider")]
        private static extern Status CreateCollider(
            [Out] out IntPtr colliderHandle,
            [In] IntPtr handle,
            [In] ref ColliderMesh mesh,
            [In] ref Transform transform
        );
    }
}