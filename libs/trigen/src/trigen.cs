using System;
using System.Runtime.InteropServices;

namespace Net.Easimer.Trigen
{
    public enum Status
    {
        OK = 0,
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
    }
}